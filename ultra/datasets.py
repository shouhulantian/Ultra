import os
import csv
import shutil
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from ultra.tasks import build_relation_graph


class GrailInductiveDataset(InMemoryDataset):

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, merge_valid_test=True):
        self.version = version
        assert version in ["v1", "v2", "v3", "v4"]

        # by default, most models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        # you'll need to delete the processed datasets then and re-run to cache a new dataset
        self.merge_valid_test = merge_valid_test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # creating fact graphs - those are graphs sent to a model, based on which we'll predict missing facts
        # also, those fact graphs will be used for filtered evaluation
        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]

        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # by default, SOTA models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        test_slice = slice(sum(num_samples[:3]), sum(num_samples)) if self.merge_valid_test else slice(sum(num_samples[:4]), sum(num_samples))
        
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice], num_relations=num_relations*2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice], num_relations=num_relations*2)
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice], num_relations=num_relations*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class FB15k237Inductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
    ]

    name = "IndFB15k237"

    def __init__(self, root, version):
        super().__init__(root, version)

class WN18RRInductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
    ]

    name = "IndWN18RR"

    def __init__(self, root, version):
        super().__init__(root, version)

class NELLInductive(GrailInductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
    ]
    name = "IndNELL"

    def __init__(self, root, version):
        super().__init__(root, version)


def FB15k237(root):
    dataset = RelLinkPredDataset(name="FB15k-237", root=root+"/fb15k237/")
    data = dataset.data
    train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type,
                        num_relations=dataset.num_relations)
    valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type,
                        num_relations=dataset.num_relations)
    test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type,
                        num_relations=dataset.num_relations)
    
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    return dataset

def WN18RR(root):
    dataset = WordNet18RR(root=root+"/wn18rr/")
    # convert wn18rr into the same format as fb15k-237
    data = dataset.data
    num_nodes = int(data.edge_index.max()) + 1
    num_relations = int(data.edge_type.max()) + 1
    edge_index = data.edge_index[:, data.train_mask]
    edge_type = data.edge_type[data.train_mask]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    edge_type = torch.cat([edge_type, edge_type + num_relations])
    train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.train_mask],
                        target_edge_type=data.edge_type[data.train_mask],
                        num_relations=num_relations*2)
    valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.val_mask],
                        target_edge_type=data.edge_type[data.val_mask],
                        num_relations=num_relations*2)
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.test_mask],
                        target_edge_type=data.edge_type[data.test_mask],
                        num_relations=num_relations*2)
    
    # build relation graphs
    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    dataset.num_relations = num_relations * 2
    return dataset


class TransductiveDataset(InMemoryDataset):

    delimiter = None
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"


class TransductiveTemporalDataset(InMemoryDataset):
    delimiter = None

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(self, quadruple_file, inv_entity_vocab={}, inv_rel_vocab={},inv_time_vocab={}):

        quadruples = []
        entity_cnt, rel_cnt, time_cnt = len(inv_entity_vocab), len(inv_rel_vocab), len(inv_time_vocab)

        with open(quadruple_file, "r", encoding="utf-8") as fin:
            for l in fin:
                t_instance = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                #print(t_instance)
                if t_instance[0] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[0]] = entity_cnt
                    entity_cnt += 1
                if t_instance[2] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[2]] = entity_cnt
                    entity_cnt += 1
                if t_instance[1] not in inv_rel_vocab:
                    inv_rel_vocab[t_instance[1]] = rel_cnt
                    rel_cnt += 1
                if t_instance[3] not in inv_time_vocab:
                    inv_time_vocab[t_instance[3]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 5 and t_instance[4] not in inv_time_vocab:
                    inv_time_vocab[t_instance[4]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 4:
                    u, r, v, t1 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]], inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]]
                    quadruples.append((u, v, r, t1))
                else:
                    u, r, v, t1, t2 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]],inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]], inv_time_vocab[t_instance[4]]
                    quadruples.append((u, v, r, t1, t2))

        return {
            "quadruples": quadruples,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": len(inv_rel_vocab),
            "num_time": len(inv_time_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
            "inv_time_vocab": inv_time_vocab
        }

    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]
        num_time = test_results['num_time']
        #self.num_relations_real = num_relations

        train_quadruples = train_results["quadruples"]
        valid_quadruples = valid_results["quadruples"]
        test_quadruples = test_results["quadruples"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_quadruples], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_quadruples])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_quadruples], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_quadruples])

        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])

        # if len(train_quadruples[0]) == 4:
        #     train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        #     valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        #     test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])
        # else:
        #     train_target_ttypes = torch.tensor([(t[3],t[4]) for t in train_quadruples])
        #     valid_target_ttypes = torch.tensor([(t[3],t[4]) for t in valid_quadruples])
        #     test_target_ttypes = torch.tensor([(t[3],t[4]) for t in test_quadruples])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_ttypes = torch.cat([train_target_ttypes, train_target_ttypes])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=valid_target_ttypes)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=test_target_ttypes)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def provide_vocab(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        return test_results["inv_entity_vocab"], test_results["inv_rel_vocab"], test_results["inv_time_vocab"]

    def __repr__(self):
        return "%s()" % (self.name)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

class TransductiveTemporalCompactDataset(InMemoryDataset):
    delimiter = None

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(self, quadruple_file, inv_entity_vocab={}, inv_rel_vocab={},inv_time_vocab={}):

        quadruples = []
        entity_cnt, rel_cnt, time_cnt = len(inv_entity_vocab), len(inv_rel_vocab), len(inv_time_vocab)

        with open(quadruple_file, "r", encoding="utf-8") as fin:
            for l in fin:
                t_instance = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                #print(t_instance)
                if t_instance[0] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[0]] = entity_cnt
                    entity_cnt += 1
                if t_instance[2] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[2]] = entity_cnt
                    entity_cnt += 1
                if t_instance[1] not in inv_rel_vocab:
                    inv_rel_vocab[t_instance[1]] = rel_cnt
                    rel_cnt += 1
                if t_instance[3] not in inv_time_vocab:
                    inv_time_vocab[t_instance[3]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 5 and t_instance[4] not in inv_time_vocab:
                    inv_time_vocab[t_instance[4]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 4:
                    u, r, v, t1 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]], inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]]
                    quadruples.append((u, v, r, t1))
                else:
                    u, r, v, t1, t2 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]],inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]], inv_time_vocab[t_instance[4]]
                    quadruples.append((u, v, r, t1, t2))

        return {
            "quadruples": quadruples,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": len(inv_rel_vocab),
            "num_time": len(inv_time_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
            "inv_time_vocab": inv_time_vocab
        }

    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]
        num_time = test_results['num_time']
        #self.num_relations_real = num_relations

        train_quadruples = train_results["quadruples"]
        valid_quadruples = valid_results["quadruples"]
        test_quadruples = test_results["quadruples"]

        compact_triples, compact_timestamps = self.provide_compact_quadruples(train_quadruples,num_time)
        compact_timestamps = torch.nn.utils.rnn.pad_sequence(compact_timestamps, batch_first=True, padding_value=-1)

        compact_edges = torch.tensor([[t[0], t[1]] for t in compact_triples], dtype=torch.long).t()
        compact_etypes = torch.tensor([t[2] for t in compact_triples])

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_quadruples], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_quadruples])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_quadruples], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_quadruples])

        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])

        # if len(train_quadruples[0]) == 4:
        #     train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        #     valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        #     test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])
        # else:
        #     train_target_ttypes = torch.tensor([(t[3],t[4]) for t in train_quadruples])
        #     valid_target_ttypes = torch.tensor([(t[3],t[4]) for t in valid_quadruples])
        #     test_target_ttypes = torch.tensor([(t[3],t[4]) for t in test_quadruples])

        # train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        # train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        # train_ttypes = torch.cat([train_target_ttypes, train_target_ttypes])

        train_edges = torch.cat([compact_edges, compact_edges.flip(0)], dim=1)
        train_etypes = torch.cat([compact_etypes, compact_etypes + num_relations])
        train_ttypes = torch.cat([compact_timestamps, compact_timestamps])
        #train_ttypes = compact_timestamps + compact_timestamps

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=valid_target_ttypes)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=test_target_ttypes)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def provide_vocab(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        return test_results["inv_entity_vocab"], test_results["inv_rel_vocab"], test_results["inv_time_vocab"]

    def provide_compact_quadruples(self, quads, num_time):
        triples_dict = {}
        for t in quads:
            if (t[0], t[1], t[2]) not in triples_dict:
                triples_dict[(t[0], t[1], t[2])] = [t[3]]
            else:
                triples_dict[(t[0], t[1], t[2])].append(t[3])
        compact_triples = list(triples_dict.keys())
        compact_timestamps = list(triples_dict.values())

        return compact_triples, compact_timestamps

    def __repr__(self):
        return "%s()" % (self.name)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

class CoDEx(TransductiveDataset):

    name = "codex"
    urls = [
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/train.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/valid.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/test.txt",
    ]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.name, self.raw_dir)
            os.rename(download_path, path)


class CoDExSmall(CoDEx):
    """
    #node: 2034
    #edge: 36543
    #relation: 42
    """
    url = "https://zenodo.org/record/4281094/files/codex-s.tar.gz"
    md5 = "63cd8186fc2aeddc154e20cf4a10087e"
    name = "codex-s"

    def __init__(self, root):
        super(CoDExSmall, self).__init__(root=root, size='s')


class CoDExMedium(CoDEx):
    """
    #node: 17050
    #edge: 206205
    #relation: 51
    """
    url = "https://zenodo.org/record/4281094/files/codex-m.tar.gz"
    md5 = "43e561cfdca1c6ad9cc2f5b1ca4add76"
    name = "codex-m"
    def __init__(self, root):
        super(CoDExMedium, self).__init__(root=root, size='m')


class CoDExLarge(CoDEx):
    """
    #node: 77951
    #edge: 612437
    #relation: 69
    """
    url = "https://zenodo.org/record/4281094/files/codex-l.tar.gz"
    md5 = "9a10f4458c4bd2b16ef9b92b677e0d71"
    name = "codex-l"
    def __init__(self, root):
        super(CoDExLarge, self).__init__(root=root, size='l')


class NELL995(TransductiveDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # training set is made out of facts+train files, so we sum up their samples to build one training graph

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/main/transductive/data/nell/test.txt",
    ]
    name = "nell995"

    @property
    def raw_file_names(self):
        return ["facts.txt", "train.txt", "valid.txt", "test.txt"]
    

    def process(self):
        train_files = self.raw_paths[:4]

        facts_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        train_results = self.load_file(train_files[1], facts_results["inv_entity_vocab"], facts_results["inv_rel_vocab"])
        valid_results = self.load_file(train_files[2], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[3], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        num_node = valid_results["num_node"]
        num_relations = train_results["num_relation"]

        train_triplets = facts_results["triplets"] + train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class ConceptNet100k(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
    ]
    name = "cnet100k"
    delimiter = "\t"

class ICEWS14(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/train.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/valid.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/test.txt",
        ]
    name = "ICEWS14"
    delimiter = "\t"

class ICEWS14Compact(TransductiveTemporalCompactDataset):

    urls = [
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/train.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/valid.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/test.txt",
        ]
    name = "ICEWS14Compact"
    delimiter = "\t"

class ICEWS14RR(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS14RR/train",
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS14RR/valid",
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS14RR/test",
        ]
    name = "ICEWS14RR"
    delimiter = "\t"

class ICEWS15RR(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS15RR/train",
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS15RR/valid",
        "https://raw.githubusercontent.com/NacyNiko/naive_rule/refs/heads/main/src_data/ICEWS15RR/test",
        ]
    name = "ICEWS15RR"
    delimiter = "\t"


class ICEWS14_sym(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/train.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews14/valid.txt",
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/icews14_symmetric_test.txt",
        ]
    name = "ICEWS14_sym"
    delimiter = "\t"

class ICEWS14_temporal_sym(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20symmetric.csv",
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20symmetric.csv",
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20symmetric.csv",
        ]
    name = "ICEWS14_temporal_sym"
    delimiter = ","

class ICEWS14_temporal_inv(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20inverse.csv",
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20inverse.csv",
        "https://raw.githubusercontent.com/shouhulantian/tkbc_time/refs/heads/main/set%20temporal%20inverse.csv",
        ]
    name = "ICEWS14_temporal_inv"
    delimiter = ","

class ICEWS0515(TransductiveTemporalDataset):

    urls = [
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/train.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/valid.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/test.txt",
        ]
    name = "ICEWS0515"
    delimiter = "\t"

class ICEWS0515Compact(TransductiveTemporalCompactDataset):

    urls = [
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/train.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/valid.txt",
        "https://raw.githubusercontent.com/soledad921/ATISE/master/icews05-15/test.txt",
        ]
    name = "ICEWS0515Compact"
    delimiter = "\t"

class GDELT(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/zhangfw123/TLT-KGE/refs/heads/main/tkbc/src_data/gdelt/train",
        "https://raw.githubusercontent.com/zhangfw123/TLT-KGE/refs/heads/main/tkbc/src_data/gdelt/valid",
        "https://raw.githubusercontent.com/zhangfw123/TLT-KGE/refs/heads/main/tkbc/src_data/gdelt/test",
    ]
    name = "GDELT"
    delimiter = "\t"

class smallpedia(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/shouhulantian/Ultra/main/datasets/smallpedia/train.txt",
        "https://raw.githubusercontent.com/shouhulantian/Ultra/main/datasets/smallpedia/valid.txt",
        "https://raw.githubusercontent.com/shouhulantian/Ultra/main/datasets/smallpedia/test.txt",
    ]
    name = "smallpedia"
    delimiter = "\t"

class Testkg(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/shouhulantian/Ultra/refs/heads/main/datasets/test_kg/train.txt",
        "https://raw.githubusercontent.com/shouhulantian/Ultra/refs/heads/main/datasets/test_kg/valid.txt",
        "https://raw.githubusercontent.com/shouhulantian/Ultra/refs/heads/main/datasets/test_kg/test.txt",
    ]
    name = "Testkg"
    delimiter = "\t"

class ICEWS14Ind(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/master/data/ICEWS14/train.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/master/data/ICEWS14/valid.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/master/data/ICEWS14/test.txt",
    ]
    name = "ICEWS14Ind"
    delimiter = "\t"

class ICEWS18Ind(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/ICEWS18/train.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/ICEWS18/valid.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/ICEWS18/test.txt",
    ]
    name = "ICEWS18Ind"
    delimiter = "\t"


class YAGOInd(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/YAGO/train.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/YAGO/valid.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/YAGO/test.txt",
    ]
    name = "YAGOInd"
    delimiter = "\t"

class GDELTInd(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/GDELT/train.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/GDELT/valid.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/GDELT/test.txt",
    ]
    name = "GDELTInd"
    delimiter = "\t"


class WIKIInd(TransductiveTemporalDataset):
    urls = [
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/WIKI/train.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/WIKI/valid.txt",
        "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/refs/heads/master/data/WIKI/test.txt",
    ]
    name = "WIKIInd"
    delimiter = "\t"

class DBpedia100k(TransductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
        ]
    name = "dbp100k"


class YAGO310(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
        ]
    name = "yago310"


class Hetionet(TransductiveDataset):

    urls = [
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        ]
    name = "hetionet"


class AristoV4(TransductiveDataset):

    url = "https://zenodo.org/record/5942560/files/aristo-v4.zip"

    name = "aristov4"
    delimiter = "\t"

    def download(self):
        download_path = download_url(self.url, self.raw_dir)
        extract_zip(download_path, self.raw_dir)
        os.unlink(download_path)
        for oldname, newname in zip(['train', 'valid', 'test'], self.raw_paths):
            os.rename(os.path.join(self.raw_dir, oldname), newname)


class SparserKG(TransductiveDataset):

    # 5 datasets based on FB/NELL/WD, introduced in https://github.com/THU-KEG/DacKGR
    # re-writing the loading function because dumps are in the format (h, t, r) while the standard is (h, r, t)

    url = "https://raw.githubusercontent.com/THU-KEG/DacKGR/master/data.zip"
    delimiter = "\t"
    base_name = "SparseKG"

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "processed")

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        for dsname in ['NELL23K', 'WD-singer', 'FB15K-237-10', 'FB15K-237-20', 'FB15K-237-50']:
            for oldname, newname in zip(['train.triples', 'dev.triples', 'test.triples'], self.raw_file_names):
                os.renames(os.path.join(base_path, "data", dsname, oldname), os.path.join(base_path, dsname, "raw", newname))
        shutil.rmtree(os.path.join(base_path, "data"))
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, v, r = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
class WDsinger(SparserKG):   
    name = "WD-singer"

class NELL23k(SparserKG):   
    name = "NELL23K"

class FB15k237_10(SparserKG):   
    name = "FB15K-237-10"

class FB15k237_20(SparserKG):   
    name = "FB15K-237-20"

class FB15k237_50(SparserKG):   
    name = "FB15K-237-50"


class InductiveDataset(InMemoryDataset):

    delimiter = None
    # some datasets (4 from Hamaguchi et al and Indigo) have validation set based off the train graph, not inference
    valid_on_inf = True  # 
    
    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, **kwargs):

        self.version = str(version)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "inf_valid.txt", "inf_test.txt"
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)


class IngramInductive(InductiveDataset):

    @property
    def raw_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "processed")
    

class FBIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/test.txt",
    ]
    name = "fb"


class WKIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/test.txt",
    ]
    name = "wk"

class NLIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/test.txt",
    ]
    name = "nl"


class ILPC2022(InductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/train.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_validation.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_test.txt",
    ]

    name = "ilpc2022"
    

class HM(InductiveDataset):
    # benchmarks from Hamaguchi et al and Indigo BM

    urls = [
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/train.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-graph.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/valid.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-fact.txt",
    ]

    name = "hm"
    versions = {
        '1k': "Hamaguchi-BM_both-1000",
        '3k': "Hamaguchi-BM_both-3000",
        '5k': "Hamaguchi-BM_both-5000",
        'indigo': "INDIGO-BM" 
    }
    # in 4 HM graphs, the validation set is based off the training graph, so we'll adjust the dataset creation accordingly
    valid_on_inf = False 

    def __init__(self, root, version, **kwargs):
        version = self.versions[version]
        super().__init__(root, version, **kwargs)

    # HM datasets are a bit weird: validation set (based off the train graph) has a few hundred new nodes, so we need a custom processing
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                          edge_type=train_fact_type, 
                          num_nodes=valid_res["num_node"],  # the only fix in this function
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class MTDEAInductive(InductiveDataset):

    valid_on_inf = False
    url = "https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip"
    base_name = "mtdea"

    def __init__(self, root, version, **kwargs):

        assert version in self.versions, f"unknown version {version} for {self.name}, available: {self.versions}"
        super().__init__(root, version, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "transductive_valid.txt", "inf_test.txt"
        ]

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        # unzip all datasets at once
        for dsname in ['FBNELL', 'Metafam', 'WikiTopics-MT1', 'WikiTopics-MT2', 'WikiTopics-MT3', 'WikiTopics-MT4']:
            cl = globals()[dsname.replace("-","")]
            versions = cl.versions
            for version in versions:
                for oldname, newname in zip(['train.txt', 'observe.txt', 'valid.txt', 'test.txt'], self.raw_file_names):
                    foldername = cl.prefix % version + "-trans" if "transductive" in newname else cl.prefix % version + "-ind"
                    os.renames(
                        os.path.join(base_path, "MTDEA_datasets", dsname, foldername, oldname), 
                        os.path.join(base_path, dsname, version, "raw", newname)
                    )
        shutil.rmtree(os.path.join(base_path, "MTDEA_datasets"))

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL and MT3:art, other datasets seem to be ok and share num_nodes/num_relations in the train/inference graph  
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))
        
        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    # special processes for MTDEA datasets for one particular fix in the validation set loading
    def process(self):
    
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class FBNELL(MTDEAInductive):

    name = "FBNELL"
    prefix = "%s"
    versions = ["FBNELL_v1"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(FBNELL, self).__init__(**kwargs)


class Metafam(MTDEAInductive):

    name = "Metafam"
    prefix = "%s"
    versions = ["Metafam"]

    def __init__(self, **kwargs):
        kwargs.pop("version")
        kwargs['version'] = self.versions[0]
        super(Metafam, self).__init__(**kwargs)


class WikiTopicsMT1(MTDEAInductive):

    name = "WikiTopics-MT1"
    prefix = "wikidata_%sv1"
    versions = ['mt', 'health', 'tax']

    def __init__(self, **kwargs):
        assert kwargs['version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)


class WikiTopicsMT2(MTDEAInductive):

    name = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['mt2', 'org', 'sci']

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)


class WikiTopicsMT3(MTDEAInductive):

    name = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['mt3', 'art', 'infra']

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)


class WikiTopicsMT4(MTDEAInductive):

    name = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['mt4', 'sci', 'health']

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)


# a joint dataset for pre-training ULTRA on several graphs
class JointDataset(InMemoryDataset):

    datasets_map = {
        'FB15k237': FB15k237,
        'WN18RR': WN18RR,
        'CoDExSmall': CoDExSmall,
        'CoDExMedium': CoDExMedium,
        'CoDExLarge': CoDExLarge,
        'NELL995': NELL995,
        'ConceptNet100k': ConceptNet100k,
        'DBpedia100k': DBpedia100k,
        'YAGO310': YAGO310,
        'AristoV4': AristoV4,
        'ICEWS14': ICEWS14,
        'GDELT':GDELT,
        'ICEWS0515':ICEWS0515,
        'Testkg': Testkg,
        'ICEWS14Compact': ICEWS14Compact,
        'ICEWS0515Compact':ICEWS0515Compact
    }

    def __init__(self, root, graphs, transform=None, pre_transform=None):


        self.graphs = [self.datasets_map[ds](root=root) for ds in graphs]
        self.num_graphs = len(graphs)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "joint", f'{self.num_graphs}g', "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    
    def process(self):
        
        train_data = [g[0] for g in self.graphs]
        valid_data = [g[1] for g in self.graphs]
        test_data = [g[2] for g in self.graphs]
        # filter_data = [
        #     Data(edge_index=g.data.target_edge_index, edge_type=g.data.target_edge_type, num_nodes=g[0].num_nodes) for g in self.graphs
        # ]

        torch.save((train_data, valid_data, test_data), self.processed_paths[0])