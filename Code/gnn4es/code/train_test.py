import torch
import argparse
import os
import numpy as np
import scipy.sparse as sp
import json
from model import GAE4ES
from torch_geometric.data import Data
from itertools import chain
from torch_geometric.utils import to_undirected, remove_self_loops
import random
import datetime
import sys
import requests
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import itertools

DATA_DIR = os.path.join(os.getcwd(), "data")  # fixed path

IN_ESDM_DIR = os.path.join(os.getcwd(), "data", "in_ESBM_benchmark_v1.2")
IN_EMBED_DIR = os.path.join(os.getcwd(), "data", "in_embed")
OUT_SUMM_DIR = os.path.join(os.getcwd(), "data", "out_summ")


def _eval_Fmeasure(summ_tids, gold_list):
    k = len(summ_tids)
    f_list = []
    for gold in gold_list:
        if len(gold) != k:
            print("gold-k:", len(gold), k)
        assert len(gold) == k  # for ESBM, not for dsFACES
        corr = len([t for t in summ_tids if t in gold])
        precision = corr / k
        recall = corr / len(gold)
        f1 = 2 * precision * recall / (precision + recall) if corr != 0 else 0
        f_list.append(f1)
    favg = np.mean(f_list)
    return favg


def gen_summ_file(
    fold,
    eid_tids_dict,
    args,
    eid,
    summ_tids,
    out_summ_dir=OUT_SUMM_DIR,
    esbm_dir=IN_ESDM_DIR,
):
    desc_tids = eid_tids_dict.get(eid)
    summ_tidxs = [desc_tids.index(tid) for tid in summ_tids]
    in_file = os.path.join(
        esbm_dir, "{}_data".format(args.dataset), str(eid), "{}_desc.nt".format(eid)
    )
    desc_lines = []
    with open(in_file, "r", encoding="utf-8") as inf:
        for triple in inf:
            if len(triple.strip()) > 0:
                desc_lines.append(triple)
    summ_lines = [desc_lines[idx] for idx in summ_tidxs]
    out_e_dir = os.path.join(
        out_summ_dir, args.dataset, "fold_{}".format(fold), str(eid)
    )
    if not os.path.isdir(out_e_dir):
        os.makedirs(out_e_dir)
    out_file = os.path.join(out_e_dir, "{}_{}.nt".format(eid, "top" + str(args.topk)))
    print("output:", out_file)
    with open(out_file, "w", encoding="utf-8") as outf:
        outf.writelines(summ_lines)


def dropout_node(desc_index, fixed_mask, negative_nodes=None):
    remaied_node = set(desc_index) - set(fixed_mask)
    edge_index = list(itertools.product(remaied_node, remaied_node))
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_index = remove_self_loops(to_undirected(edge_index))[0]
    eval_index = []
    for item in fixed_mask:
        eval_index.append([(item, i) for i in remaied_node])
    eval_index = list(itertools.chain(*eval_index))
    eval_index = torch.tensor(eval_index).t().contiguous()
    if negative_nodes is None:
        return edge_index, eval_index
    else:
        neg_eval_index = []
        for item in negative_nodes:
            neg_eval_index.append([(item, i) for i in remaied_node])
        neg_eval_index = list(itertools.chain(*neg_eval_index))
        neg_eval_index = torch.tensor(neg_eval_index).t().contiguous()
        return edge_index, eval_index, neg_eval_index


def K_means(data, k, max_iter=100, tol=1e-4):
    # 初始化中心点（使用数据中的前 k 个点）
    centroids = data[:k]
    for _ in range(max_iter):
        # 计算每个点到各个中心点的距离
        distances = torch.cdist(data, centroids)
        # 将每个点分配给最近的中心点
        cluster_assignments = torch.argmin(distances, dim=1)
        # 更新中心点
        new_centroids = torch.empty_like(centroids)
        for i in range(k):
            cluster_data = data[cluster_assignments == i]
            if len(cluster_data) == 0:
                # 如果聚类没有数据点，重新分配一个随机质心
                new_centroids[i] = data[torch.randint(len(data), (1,))][0]
            else:
                new_centroids[i] = cluster_data.mean(dim=0)
        # 检查收敛
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, cluster_assignments


def got_dataset_centroid(dataset, model_name, K):
    features_path = os.path.join(
        IN_EMBED_DIR, "{}_{}_vec.npz".format(dataset, model_name)
    )
    features = np.load(features_path, allow_pickle=True)
    features_dict = eval(str(features["pvembedding_ftaw"]))
    features_dict = {key: torch.tensor(value) for key, value in features_dict.items()}
    embeddings = torch.stack(list(features_dict.values()))
    centroids, cluster_assignments = K_means(embeddings, K)
    id_to_cluster = {}
    id_to_cluster = {
        idx: cluster_label.item()
        for idx, cluster_label in zip(features_dict.keys(), cluster_assignments)
    }
    return centroids, id_to_cluster


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.eid2smset_dict = self.get_eid2smset_dict()
        if self.args.dataset == "faces":
            label_vec_path = os.path.join(
                DATA_DIR,
                "in_embed",
                "faces_label_{}_vec.npz".format(self.args.pre_trained_model),
            )
        else:
            label_vec_path = os.path.join(
                DATA_DIR,
                "in_embed",
                "label_{}_vec.npz".format(self.args.pre_trained_model),
            )
        self.label_vec_matrix = np.load(label_vec_path, allow_pickle=True)["arr_0"]
        self.eid2did_dict = self.get_eid2descset_dict()
        self.eid_glods_dict = self.get_glods_dict()
        self.property_frequency = self.get_property_frequency()
        self.centroids, self.id_to_cluster = got_dataset_centroid(
            self.args.dataset, self.args.pre_trained_model, K=5
        )

    def _do_train(self, fold):
        # train_graph_list = self.construct_Graph_Data(fold, "train")
        valid_graph_list = self.construct_Test_data(fold, "valid")
        input_size = len(self.label_vec_matrix[0]) * 2
        model = GAE4ES(
            self.args, input_size, self.args.hidden_size, self.args.hidden_size
        ).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        stop_valid_score = None
        train_graph_list = []
        for i in range(self.args.sub_graph_num, 0, -1):
            train_graph_list.extend(
                self.construct_Graph_Data(fold, "train", mask_num=i)
            )
        for epoch in range(self.args.epochs):
            total_loss = 0
            total_triple_loss = 0
            total_link_loss = 0
            total_cali_loss = 0
            model.train()
            for graph in tqdm(train_graph_list):
                optimizer.zero_grad()
                loss, loss_term = model(graph)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_triple_loss += loss_term[0].item()
                total_link_loss += loss_term[1].item()
                total_cali_loss += loss_term[2].item()
            total_loss = total_loss / len(train_graph_list)
            total_triple_loss /= len(train_graph_list)
            total_link_loss /= len(train_graph_list)
            total_cali_loss /= len(train_graph_list)
            # writer.add_scalar('train_loss',total_loss,epoch+1)
            print(
                "train loss in epoch {} ,triple_loss is {}, link_loss is {}, cali_loss is {} ,total_loss is :{}".format(
                    epoch + 1,
                    total_triple_loss,
                    total_link_loss,
                    total_cali_loss,
                    total_loss,
                )
            )
            if epoch > 0 and epoch % self.args.valid_epoch == 0:
                valid_score = self._do_test(
                    fold, "valid", valid_graph_list, model=model
                )
                if stop_valid_score == None or valid_score > stop_valid_score:
                    stop_valid_score = valid_score
                    if self.args.model_save_dir != None:
                        model_save_path = os.path.join(
                            self.args.model_save_dir,
                            "{}/{}/{}".format(
                                self.args.dataset,
                                "fold" + str(fold),
                                str(self.args.topk),
                            ),
                        )
                        if os.path.exists(model_save_path) is not True:
                            os.makedirs(model_save_path)
                        torch.save(model.state_dict(), model_save_path + "/model.pth")
                        print(
                            "save model in epoch {},model in {}".format(
                                epoch + 1, model_save_path
                            )
                        )
                print("valid score in epoch {} : {}".format(epoch + 1, valid_score))
                # writer.add_scalar('valid_loss',valid_total_loss,epoch+1)

    def _do_test(self, fold, phase, test_data_list=None, model=None):
        if phase == "test":
            test_data_list = self.construct_Test_data(fold, "test")
            model_save_path = os.path.join(
                self.args.model_save_dir,
                "{}/{}/{}".format(
                    self.args.dataset, "fold" + str(fold), str(self.args.topk)
                ),
            )
            model = GAE4ES(
                self.args,
                test_data_list[0].x.shape[1],
                self.args.hidden_size,
                self.args.hidden_size,
                training=False,
            ).to(self.args.device)
            model.load_state_dict(torch.load(model_save_path + "/model.pth"))
            print("model load from {} success".format(model_save_path))
        model.eval()
        result_json = {}
        favg_list = []
        for test_data in tqdm(test_data_list):
            gene_topk = []
            desc_idx2id_dict = test_data.desc_idx2id_dict
            dest_node = []
            edge_index_row = []
            edge_index_col = []
            label_index = test_data.x.shape[0] - 1
            while len(gene_topk) < self.args.topk:
                compatibility_score = torch.zeros(label_index, device=self.args.device)
                edge_index = torch.stack(
                    [
                        torch.tensor(edge_index_row, dtype=torch.long),
                        torch.tensor(edge_index_col, dtype=torch.long),
                    ]
                )
                edge_index = to_undirected(edge_index)
                graph_data = Data(test_data.x, edge_index=edge_index)
                if len(dest_node) is 0:
                    graph_data = graph_data.to(self.args.device)
                    x = model.embedding_transfer(graph_data.x)
                    label_embedding = x[-1]
                    sim_score = torch.matmul(x[:-1], label_embedding)
                    temp_index = sim_score.argmax().item()
                    dest_node.append(temp_index)
                    gene_topk.append(desc_idx2id_dict[temp_index])
                    continue
                edge_label_index = [[], []]
                for src in range(label_index):
                    for dest in dest_node:
                        edge_label_index[0].append(src)
                        edge_label_index[1].append(dest)
                eval_index = torch.tensor(edge_label_index, dtype=torch.long)
                graph_data.eval_edge = eval_index
                graph_data = graph_data.to(self.args.device)
                sim_score, link_score = model(graph_data)
                link_score = link_score.view(label_index, -1)
                steps = len(dest_node) + 1
                dest_node_tensor = torch.tensor(
                    dest_node, dtype=torch.long, device=self.args.device
                )
                dest_mask = (
                    torch.nn.functional.one_hot(
                        dest_node_tensor, num_classes=label_index
                    ).sum(dim=0)
                    > 0
                )
                src_mask = ~dest_mask
                compatibility_score[src_mask] += link_score.sum(-1)[src_mask]
                compatibility_score[dest_mask] = float("-inf")
                compatibility_score = compatibility_score / steps
                max_score, _ = compatibility_score[
                    compatibility_score != float("-inf")
                ].max(dim=0)
                min_score, _ = compatibility_score[
                    compatibility_score != float("-inf")
                ].min(dim=0)
                compatibility_score[compatibility_score != float("-inf")] = (
                    compatibility_score[compatibility_score != float("-inf")]
                    - min_score
                ) / (max_score - min_score)
                sim_score = sim_score - sim_score.min() / (
                    sim_score.max() - sim_score.min()
                )
                temp_index = (compatibility_score + sim_score).argmax().item()
                temp_index = compatibility_score.argmax().item()
                if len(dest_node) > 0:
                    edge_index_row.extend([temp_index] * len(dest_node))
                    edge_index_col.extend(dest_node)
                dest_node.append(temp_index)
                gene_topk.append(desc_idx2id_dict[temp_index])
            if phase == "test" and self.args.out_to_files:
                gen_summ_file(
                    fold,
                    self.eid2did_dict,
                    self.args,
                    test_data.entity,
                    gene_topk,
                    out_summ_dir=OUT_SUMM_DIR + "/" + self.args.backbone,
                    esbm_dir=IN_ESDM_DIR,
                )
            result_json[test_data.entity] = gene_topk
            # check generate set is available
            desc_ids = self.eid2did_dict[test_data.entity]
            if set(gene_topk) <= set(desc_ids) is False:
                sys.exit()
            golds = self.eid_glods_dict[test_data.entity]
            favg = _eval_Fmeasure(gene_topk, golds)
            favg_list.append(favg)
            if phase == "test":
                print(
                    "generate entity {} summay set is {}, got f1 score is {}".format(
                        test_data.entity, gene_topk, favg
                    )
                )
        test_favg = np.mean(favg_list)
        print("{} fold {} got favg score is {} ".format(phase, fold, test_favg))
        return test_favg

    def get_glods_dict(self):
        # load data from file
        eid_glods_dict = {}
        eid_glods_path = os.path.join(
            DATA_DIR,
            "in_embed",
            "{}_egolds_top{}.txt".format(self.args.dataset, self.args.topk),
        )
        with open(eid_glods_path) as f:
            for line in f:
                item = line.strip().split("\t")
                eid = int(item[0])
                smset_list = eval(item[1])
                eid_glods_dict[eid] = smset_list
        return eid_glods_dict

    def get_eid2descset_dict(self):
        eid2did_dict = {}
        eid2did_path = os.path.join(
            DATA_DIR, "in_embed", "{}_tids.txt".format(self.args.dataset)
        )
        with open(eid2did_path) as f:
            for line in f:
                item = line.strip().split("\t")
                eid = int(item[0])
                did_list = eval(item[1])
                eid2did_dict[eid] = did_list
        return eid2did_dict

    def get_eid2smset_dict(self):
        # load data from file
        eid2smset_dict = {}
        eid2smset_path = os.path.join(
            DATA_DIR,
            "in_embed",
            "{}_egolds_top{}.txt".format(self.args.dataset, self.args.topk),
        )
        with open(eid2smset_path) as f:
            for line in f:
                item = line.strip().split("\t")
                eid = int(item[0])
                smset_list = eval(item[1])
                smset = set()
                for sms in smset_list:
                    for i in sms:
                        smset.add(i)
                eid2smset_dict[eid] = smset
        return eid2smset_dict

    def get_max_len(self):
        max_len = 0
        tids_file_path = os.path.join(
            DATA_DIR, "in_embed", "{}_tids.txt".format(self.args.dataset)
        )
        with open(tids_file_path) as f:
            for line in f:
                item = line.strip().split("\t")
                eid = item[0]
                tids = eval(str(item[1]))
                if len(tids) > max_len:
                    max_len = len(tids)
        return max_len

    def get_property_frequency(self):
        property_frequency_path = os.path.join(
            DATA_DIR, "in_embed", "property_frequency_{}.json".format(self.args.dataset)
        )
        property_frequency_dict = json.load(open(property_frequency_path))
        return property_frequency_dict

    def construct_Test_data(self, fold, phase):
        fold_path = os.path.join(
            DATA_DIR,
            "in_ESBM_benchmark_v1.2",
            "{}_split".format(self.args.dataset),
            "Fold{}".format(fold),
        )
        top_path = os.path.join(fold_path, "top{}".format(self.args.topk))
        # got test entities and their descriptions
        test_eids_file = fold_path + "/{}_eids.txt".format(phase)
        with open(test_eids_file) as f:
            test_eids_list = eval(f.readline())
        test_eid2did_dict = {item: self.eid2did_dict[item] for item in test_eids_list}
        # got test features matrix tensor
        test_features_file = top_path + "/{}_features_{}.npz".format(
            self.args.pre_trained_model, phase
        )
        test_id2idx_file = top_path + "/id2idx_{}.json".format(phase)
        test_features_matrix = np.load(test_features_file, allow_pickle=True)["arr_0"]
        with open(test_id2idx_file) as f:
            test_id2idx = json.load(f)
        # prepare graph data to test
        desc_embed_size = len(test_features_matrix[0])
        ent_embed_size = len(self.label_vec_matrix[0])
        test_data_list = []
        for entity in test_eids_list:
            desc_set = test_eid2did_dict[entity]
            entity_label_feature_tensor = torch.tensor(
                self.label_vec_matrix[entity - 1]
            )
            pad_length = (desc_embed_size - ent_embed_size) // 2
            ed_tensor = torch.zeros(len(desc_set) + 1, desc_embed_size).to(
                self.args.device
            )
            desc_idx2id_dict = {}
            padding_label_tensor = torch.nn.functional.pad(
                entity_label_feature_tensor, (pad_length, pad_length), "constant", 0
            )
            ed_tensor[-1] = padding_label_tensor
            des_index = 0
            for des in desc_set:
                ed_tensor[des_index] = torch.tensor(
                    test_features_matrix[test_id2idx[str(des)]]
                )
                desc_idx2id_dict[des_index] = des
                des_index += 1
            graph_data = Data(x=ed_tensor, edge_index=None)
            graph_data.desc_idx2id_dict = desc_idx2id_dict
            graph_data.entity = entity
            test_data_list.append(graph_data)
        return test_data_list

    def construct_Graph_Data(self, fold, phase, mask_num):
        fold_path = os.path.join(
            DATA_DIR,
            "in_ESBM_benchmark_v1.2",
            "{}_split".format(self.args.dataset),
            "Fold{}".format(fold),
        )
        top_path = os.path.join(fold_path, "top{}".format(self.args.topk))
        id2idx_file_path = top_path + "/id2idx_{}.json".format(phase)
        with open(fold_path + "/{}_eids.txt".format(phase)) as f:
            line = f.readline()
            eids_list = eval(line)
        with open(id2idx_file_path) as f:
            id2idx_dict = json.load(f)
        frequency_file = top_path + "/frequency_{}.json".format(phase)
        with open(frequency_file) as f:
            frequency_dict = json.load(f)
        features_file = top_path + "/{}_features_{}.npz".format(
            self.args.pre_trained_model, phase
        )
        features_tensor = torch.tensor(
            np.load(features_file, allow_pickle=True)["arr_0"]
        )
        fold_data_list = []
        for eid in eids_list:
            desc_list = self.eid2did_dict[eid]
            eglods_list = self.eid_glods_dict[eid]
            sm_list = self.eid2smset_dict[eid]
            negative_list = set(desc_list) - sm_list
            label_vec_tensor = torch.tensor(self.label_vec_matrix[eid - 1])
            desc_tensor = torch.empty(len(desc_list), features_tensor.shape[1])
            for i in range(len(desc_list)):
                desc_tensor[i] = features_tensor[id2idx_dict[str(desc_list[i])]]
            padd_len = (desc_tensor.shape[1] - len(label_vec_tensor)) // 2
            padding_label_tensor = torch.nn.functional.pad(
                label_vec_tensor, (padd_len, padd_len), "constant", 0
            )
            ed_tensor = torch.cat(
                (desc_tensor, padding_label_tensor.unsqueeze(0)), dim=0
            )
            label_index = ed_tensor.shape[0] - 1
            temp_id2idx = {}
            temp_idx2id = {}
            index_to_cluster_id = {}
            for i in range(desc_tensor.shape[0]):
                temp_id2idx[desc_list[i]] = i
                index_to_cluster_id[i] = self.id_to_cluster[desc_list[i]]
                temp_idx2id[i] = desc_list[i]
            negative_index = torch.tensor(
                [temp_id2idx[i] for i in negative_list], dtype=torch.long
            )
            postive_index = torch.tensor(
                [temp_id2idx[i] for i in sm_list], dtype=torch.long
            )
            if len(postive_index) < len(negative_index):
                repeat_times = len(negative_index) // len(postive_index)
                postive_index = torch.cat(
                    (postive_index, postive_index.repeat(repeat_times)), dim=0
                )[: len(negative_index)]
            elif len(postive_index) > len(negative_index):
                repeat_times = len(postive_index) // len(negative_index)
                negative_index = torch.cat(
                    (negative_index, negative_index.repeat(repeat_times)), dim=0
                )[: len(postive_index)]
            assert len(negative_index) == len(postive_index)
            graph_list = []
            ed_link_edge = torch.tensor(
                [[label_index, temp_id2idx[desc_id]] for desc_id in desc_list],
                dtype=torch.long,
            ).transpose(0, 1)
            for egold in eglods_list:
                desc_index = [temp_id2idx[desc_id] for desc_id in egold]
                random_sample_node = random.sample(desc_index, k=mask_num)
                if self.args.neg_sample:
                    neg_sample_node = random.sample(negative_index.tolist(), k=mask_num)
                    edge_index, eval_edge, neg_eval_edge = dropout_node(
                        desc_index, random_sample_node, neg_sample_node
                    )
                    neg_eval_label = [0] * len(neg_eval_edge[0])
                else:
                    edge_index, eval_edge = dropout_node(
                        desc_index, fixed_mask=random_sample_node
                    )
                graph_data = Data(x=None, edge_index=edge_index)
                eval_label = []
                for i in range(len(eval_edge[0])):
                    if self.args.dataset == "faces":
                        eval_label.append(
                            (
                                frequency_dict[str(temp_idx2id[eval_edge[0][i].item()])]
                                + frequency_dict[
                                    str(temp_idx2id[eval_edge[1][i].item()])
                                ]
                            )
                            / 2
                        )
                    else:
                        eval_label.append(
                            (
                                frequency_dict[str(temp_idx2id[eval_edge[0][i].item()])]
                                + frequency_dict[
                                    str(temp_idx2id[eval_edge[1][i].item()])
                                ]
                            )
                            / 12
                        )
                if self.args.neg_sample:
                    # eval_edge.extend(neg_eval_edge)
                    eval_edge = torch.cat((eval_edge, neg_eval_edge), dim=1)
                    eval_label.extend(neg_eval_label)
                graph_data.eval_label = torch.tensor(eval_label)
                graph_data.eval_edge = eval_edge
                graph_data = graph_data.to(self.args.device)
                graph_data.desc_index = desc_index
                graph_list.append(graph_data)
            graph = Data(x=ed_tensor, edge_index=None)
            graph.ed_edge_index = ed_link_edge
            graph.link_graph_list = graph_list
            graph.postive_index = postive_index
            graph.negative_index = negative_index
            graph = graph.to(self.args.device)
            graph.centroids = self.centroids
            graph.index_to_cluster_id = index_to_cluster_id
            fold_data_list.append(graph)
        return fold_data_list


def setup_seed(seed=3047):
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(seed=2023)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="faces", choices=["dbpedia", "lmdb", "faces"]
    )
    parser.add_argument("--topk", default=5, type=int, choices=[5, 10])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--valid_epoch", default=1, type=int)
    parser.add_argument("--backbone", default="DisenGCN", choices=["GCN", "DisenGCN"])
    parser.add_argument(
        "--model_save_dir",
        default=os.path.join(os.getcwd(), "data"),
        type=str,
    )
    parser.add_argument("--hidden_size", default=64)
    parser.add_argument("--pre_trained_model", default="clip-vit-base-patch32")
    parser.add_argument("--out_to_files", default=False, type=bool)
    parser.add_argument("--triplet_margin", default=1, type=float)
    parser.add_argument("--dcl_output_layers", default=4, type=int)
    parser.add_argument("--sub_graph_num", default=1, type=int)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--neg_sample", default=False, type=bool)
    args = parser.parse_args()
    out_file = os.path.join(args.model_save_dir, "results.txt")
    log_dir = os.path.join(args.model_save_dir, "log")

    # args.model_save_dir = os.path.join(args.model_save_dir,'DisenGCN','out_model')
    # trainer = Trainer(args)
    # trainer._do_test(0,phase='test')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    result_dict = {}
    with open(out_file, "a+") as f:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if args.backbone == "GCN":
            args.model_save_dir = os.path.join(args.model_save_dir, "GCN", "out_model")
        else:
            args.model_save_dir = os.path.join(
                args.model_save_dir, "DisenGCN", "out_model"
            )
        temp_result = {}
        print(args)
        result = []
        trainer = Trainer(args)
        for fold in range(5):
            print(
                "=========================== train fold {} ===============================".format(
                    fold
                )
            )
            trainer._do_train(fold)
            print(
                "=========================== test fold {} ================================".format(
                    fold
                )
            )
            test_favg = trainer._do_test(fold, phase="test")
            result.append(test_favg)
        temp_result[args.topk] = (np.mean(result), result)
        print(
            "In dataset {},topk{} model performance {}, every score is {} \n".format(
                args.dataset, args.topk, np.mean(result), result
            )
        )
        result_dict[args.dataset] = temp_result
        print(
            "\ntest on {}\n trainning config is {} \n the final result is {}".format(
                start_time, args, result_dict
            ),
            file=f,
        )
