import os
import argparse
import numpy as np
import shutil

# Use dynamic path based on current working directory
IN_ESBM_DIR = os.path.join(os.getcwd(), 'data')
parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=2, type=int, choices=[0, 1, 2, 3, 4])
parser.add_argument("--topk", default=5, type=int, choices=[5, 10])
parser.add_argument(
    "--dataset", default="lmdb", type=str, choices=["dbpedia", "lmdb", "faces"]
)
parser.add_argument("--phase", default="train", choices=["train", "valid"])
args = parser.parse_args()
fold_path = os.path.join(
    IN_ESBM_DIR,
    "in_ESBM_benchmark_v1.2",
    "{}_split".format(args.dataset),
    "Fold{}".format(args.fold),
)


def del_file(path):
    if not os.listdir(path):
        print("rm path error")
    else:
        for i in os.listdir(path):
            # print(i)
            if i not in ["train.txt", "valid.txt", "test.txt"]:
                file_path = os.path.join(path, i)
                print(file_path)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    del_file(file_path)
                    shutil.rmtree(file_path)


if __name__ == "__main__":
    del_file(fold_path)
