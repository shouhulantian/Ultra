import os
import sys
import ast
import copy
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ultra import models, datasets


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device

def generate_splits(
        self,
        full_data: Dict[str, Any],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    r"""Generates train, validation, and test splits from the full dataset
    Args:
        full_data: dictionary containing the full dataset
        val_ratio: ratio of validation data
        test_ratio: ratio of test data
    Returns:
        train_data: dictionary containing the training dataset
        val_data: dictionary containing the validation dataset
        test_data: dictionary containing the test dataset
    """
    #my_data = np.genfromtxt('my_file.csv', delimiter=',')
    val_time, test_time = list(
        np.quantile(
            full_data["timestamps"],
            [(1 - val_ratio - test_ratio), (1 - test_ratio)],
        )
    )
    timestamps = full_data["timestamps"]

    train_mask = timestamps <= val_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    return train_mask, val_mask, test_mask

def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")

    ds_cls = getattr(datasets, cls)
    dataset = ds_cls(**data_config)

    if get_rank() == 0:
        logger.warning("%s dataset" % (cls if "version" not in cfg.dataset else f'{cls}({cfg.dataset.version})'))
        if cls != "JointDataset":
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                        (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                            dataset[2].target_edge_index.shape[1]))
        else:
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                           (sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                            ))

    return dataset

def restructure_pickle_file(pickle_file: dict, num_rels: int) -> list:
        """
        Restructure the pickle format to be able to use the functions in RE-GCN implementations.
        The main idea is to use them as tensors so itspeeds up the computations
        :param pickle_file:
        :param num_rels:
        :return:
        """

        test_triples, final_scores, timesteps = [], [], []
        for query, scores in pickle_file.items():
            timestep = int(query.split('_')[-1])
            timesteps.append(timestep)
        timestepsuni = np.unique(timesteps)  # list with unique timestamps

        timestepsdict_triples = {}  # dict to be filled with keys: timestep, values: list of all triples for that timestep
        timestepsdict_scores = {}  # dict to be filled with keys: timestep, values: list of all scores for that timestep

        for query, scores in pickle_file.items():
            timestep = int(query.split('_')[-1])
            triple = query.split('_')[:-1]
            triple = np.array([int(elem.replace('xxx', '')) if 'xxx' in elem else elem for elem in triple],
                              dtype='int32')
            if query.startswith('xxx'):  # then it was subject prediction -
                triple = triple[np.argsort([2, 1, 0])]  # so we have to turn around the order
                triple[1] = triple[1] + num_rels  # and the relation id has to be original+num_rels to indicate it was
                # other way round

            if timestep in timestepsdict_triples:
                timestepsdict_triples[timestep].append(torch.tensor(triple))
                timestepsdict_scores[timestep].append(torch.tensor(scores[0]))
            else:
                timestepsdict_triples[timestep] = [torch.tensor(triple)]
                timestepsdict_scores[timestep] = [torch.tensor(scores[0])]

        for t in np.sort(list(timestepsdict_triples.keys())):
            test_triples.append(torch.stack(timestepsdict_triples[t]))
            final_scores.append(torch.stack(timestepsdict_scores[t]))

        return timestepsuni, test_triples, final_scores

def tranform_reccurrency2ultra(pickle_file, num_rels):
    ts, triples, scores = restructure_pickle_file(pickle_file, num_rels)
    ent_vocab, rel_vocab, time_vocab = datasets.ICEWS14Ind.provide_vocab()
    ts_convert , triples_convert = [], []
    for t in ts:
        t_new = time_vocab[t]
        ts_convert.append(t_new)

    for snapshot in triples:
        transformed_tensor = np.empty_like(snapshot)

        for i in range(snapshot.shape[0]):
            head_entity_idx = snapshot[i, 0]
            relation_idx = snapshot[i, 1]
            tail_entity_idx = snapshot[i, 2]

            head_entity_name = ent_vocab[head_entity_idx]
            relation_name = rel_vocab[relation_idx]
            tail_entity_name = ent_vocab[tail_entity_idx]

            # Transform to vocab indices
            head_entity_vocab_idx = list(ent_vocab.keys())[list(ent_vocab.values()).index(head_entity_name)]
            relation_vocab_idx = list(rel_vocab.keys())[list(rel_vocab.values()).index(relation_name)]
            tail_entity_vocab_idx = list(ent_vocab.keys())[list(ent_vocab.values()).index(tail_entity_name)]

            transformed_tensor[i, 0] = head_entity_vocab_idx
            transformed_tensor[i, 1] = relation_vocab_idx
            transformed_tensor[i, 2] = tail_entity_vocab_idx
        triples_convert.append(transformed_tensor)

    return ts_convert, triples_convert,scores


