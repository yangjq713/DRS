import os
import argparse
import numpy as np
import json
from scipy.sparse import lil_matrix, save_npz, csr_matrix

# Use dynamic path based on current working directory
IN_ESBM_DIR = os.path.join(os.getcwd(), 'data')

# creat fold-n train,test,valid matrix and entity_id to embed_index map

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int, choices=[0, 1, 2, 3, 4])
parser.add_argument("--topk", default=5, type=int, choices=[5, 10])
parser.add_argument(
    "--dataset", default="dbpedia", type=str, choices=["dbpedia", "lmdb", "faces"]
)
parser.add_argument("--phase", default="test")
parser.add_argument("--model_name", default="bert-base-uncased")
args = parser.parse_args()
fold_path = os.path.join(
    IN_ESBM_DIR,
    "in_ESBM_benchmark_v1.2",
    "{}_split".format(args.dataset),
    "Fold{}".format(args.fold),
)
save_path = os.path.join(fold_path, "top{}".format(args.topk))
model_name = args.model_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

eid_list = []
with open(fold_path + "/{}.txt".format(args.phase)) as f:
    for line in f:
        item = line.strip().split("\t")
        eid = int(item[0])
        eid_list.append(eid)
print("got {} entity list: {}".format(args.phase, eid_list))
eid_list_file = fold_path + "/{}_eids.txt".format(args.phase)
with open(eid_list_file, "w") as f:
    f.writelines(str(eid_list))

eid2did_dict = {}
eid2did_path = os.path.join(IN_ESBM_DIR, "in_embed", "{}_tids.txt".format(args.dataset))
with open(eid2did_path) as f:
    for line in f:
        item = line.strip().split("\t")
        eid = int(item[0])
        did_list = eval(item[1])
        eid2did_dict[eid] = did_list

# load pre-trained features
features_path = os.path.join(
    IN_ESBM_DIR, "in_embed", "{}_{}_vec.npz".format(args.dataset, model_name)
)
features = np.load(features_path, allow_pickle=True)
features_dict = eval(str(features["pvembedding_ftaw"]))
desc_features = []
index = 0
id2idx = {}
for eid in eid_list:
    descset = eid2did_dict[eid]
    for item in descset:
        item = int(item)
        item_feature = features_dict[item]
        desc_features.append(item_feature)
        id2idx[item] = index
        index += 1

# save desc item id to feature matrix index
map_file = save_path + "/id2idx_{}.json".format(args.phase)
with open(map_file, "w") as f:
    json.dump(id2idx, f)

assert len(desc_features) == index

save_feat_file = save_path + "/{}_features_{}".format(model_name, args.phase)
np.savez(save_feat_file, np.array(desc_features))
