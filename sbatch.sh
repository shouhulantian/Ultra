#!/bin/sh
#python3 script/run.py -c config/transductive/inference.yaml  --bpe null --gpus [2] --dataset ICEWS14  --epochs 10 --ckpt /workspace/git/ULTRA/output/Ultra/ICEWS0515/model_epoch_1.pth -c config/transductive/0shot_inference.yaml --dataset ICEWS0515
#python3 script/run.py -c config/transductive/inference.yaml  --bpe null --dataset ICEWS14  --epochs 10 --ckpt null --gpus null
#/workspace/git/Ultra/config/transductive/0shot_inference.yaml
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/transductive/0shot_inference.yaml  --bpe null --dataset ICEWS0515  --epochs 10 --ckpt null --gpus null