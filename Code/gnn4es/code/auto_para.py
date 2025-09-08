import itertools
import subprocess
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", type=str, default="dbpedia")
parser.add_argument("--pre_trained_model", type=str, default="clip-vit-base-patch32")

args = parser.parse_args()
backbone = "DisenGCN"
device = "cuda:{}".format(args.device)
valid_epoch = 1
pre_trained_model = args.pre_trained_model

if args.dataset == "dbpedia":
    lrs = [0.001, 0.002]
    triplet_margin = [0.1, 0.105]
    sub_graph_nums = [1, 2]
    alphas = [0.005, 0.01, 0.015]
elif args.dataset == "lmdb":
    lrs = [0.001, 0.002]
    triplet_margin = [0.1, 0.105]
    sub_graph_nums = [2, 3, 4]
    alphas = [0.005, 0.01, 0.015]
elif args.dataset == "faces":
    lrs = [0.001, 0.002]
    triplet_margin = [0.1, 0.105]
    sub_graph_nums = [2, 3]
    alphas = [0.005, 0.01, 0.015]

K = 5
# sub_graph_nums = [3,4]

# 生成不同的参数组合
params = list(itertools.product(lrs, sub_graph_nums, triplet_margin, alphas))
# 循环不同的参数组合
for param in params:
    lr, subs, margin, alpha = param
    for topk in [5, 10]:
        cmd_args = f"--lr {lr} --dataset {args.dataset} --backbone {backbone} --device {device} --valid_epoch {valid_epoch} --topk {topk} --pre_trained_model {pre_trained_model} \
                       --triplet_margin {margin} --sub_graph_num {subs}  --alpha {alpha}"
        # 在命令行中运行训练脚本
        process = subprocess.Popen(f"python code/train_test.py {cmd_args}", shell=True)
        process.wait()
        time.sleep(10)
