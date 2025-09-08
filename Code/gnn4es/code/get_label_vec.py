# prepare the embedding of entity_label
import torch
import numpy as np

# import fasttext
import os
import re
import io
from os import path
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPTextModel,
    GPT2Tokenizer,
    GPT2Model,
    BertTokenizer,
    BertModel,
)
import json
import argparse
import fasttext

IN_ESBM_DIR = os.path.join(os.getcwd(), "data")
# model_name = 'FastText'
# model_name = 'clip-vit-base-patch32'
# model_name = 'bert-large-uncased'
model = None
tokenizer = None
embed = None


def parser(f):
    triples = list()
    for i, triple in enumerate(f):
        # extract subject
        sub = triple.strip().replace("<", "").split(">")[0]
        sub = sub[sub.rfind("/") + 1 :]
        # extract content from "content"
        if '"' in sub:
            pattern = re.compile('"(.*)"')
            try:
                sub_new = pattern.findall(sub)[0]
            except IndexError:
                # like "United States/Australian victory"
                sub = sub.replace('"', "").strip()
                sub_new = sub
        # extract content from ":content"
        elif ":" in sub:
            pattern = re.compile(":(.*)")
            sub_new = pattern.findall(sub)[0]
        else:
            sub_new = sub
        sub_new = sub_new.replace(" ", "")

        # extract object
        obj = triple.strip().replace("<", "").split(">")[2]
        # fix extract content form "content\"
        if obj.rfind("/") + 1 == len(obj):
            obj = obj[:-1]
        obj = obj[obj.rfind("/") + 1 :]
        # extract content from "content"
        if '"' in obj:
            pattern = re.compile('"(.*)"')
            try:
                obj_new = pattern.findall(obj)[0]
            except IndexError:
                # like "United States/Australian victory"
                obj = obj.replace('"', "").strip()
                obj_new = obj
        # extract content from ":content"
        elif ":" in obj:
            pattern = re.compile(":(.*)")
            obj_new = pattern.findall(obj)[0]
        else:
            obj_new = obj
        obj_new = obj_new.replace(" ", "")
        if obj_new == "":
            obj_new = "UNK"

        # extract predicate
        pred = triple.strip().replace("<", "").split(">")[1]
        pred = pred[pred.rfind("/") + 1 :]
        if "#" in pred:
            pattern = re.compile("#(.*)")
            pred_new = pattern.findall(pred)[0]
        elif ":" in pred:
            pattern = re.compile(":(.*)")
            pred_new = pattern.findall(pred)[0]
        else:
            pred_new = pred
        pred_new = pred_new.replace(" ", "")
        if not (sub_new == "" or pred_new == "" or obj_new == ""):
            triple_tuple = (
                i,
                sub,
                pred,
                obj,
                sub_new.replace(" ", ""),
                pred_new.replace(" ", ""),
                obj_new.replace(" ", ""),
            )
            triples.append(triple_tuple)
        else:
            print(triple)
    return triples


def prepare_data(db_path, num):
    with open(
        path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8"
    ) as f:
        triples = parser(f)
    return triples


def get_eid2did_dict(dataset):
    eid2did_dict = {}
    eid2did_path = os.path.join(IN_ESBM_DIR, "in_embed", "{}_tids.txt".format(dataset))
    with open(eid2did_path) as f:
        for line in f:
            item = line.strip().split("\t")
            eid = int(item[0])
            did_list = eval(item[1])
            eid2did_dict[eid] = did_list
    return eid2did_dict


def get_label_vector(dataset):
    eid_label_dict = {}
    index = 0
    # ESBM
    if dataset == "ESBM":
        with open(IN_ESBM_DIR + "/in_ESBM_benchmark_v1.2/elist.txt") as f:
            for line in f:
                if index is 0:
                    index += 1
                    continue
                item = line.strip().split("\t")
                eid = int(item[0])
                elabel = item[4]
                eid_label_dict[eid] = elabel
        # convert entity label to word vec , entity index : label_vec, average vec_matrix: scale the matrix to [-1,1]
        vec_matrix = []
        for item in eid_label_dict.values():
            words_vec = get_avg_vec(item)
            vec_matrix.append(words_vec)
        vec_ndarray = np.array(vec_matrix)
        print("label vec shape is {}".format(vec_ndarray.shape[1]))
        np.savez(
            IN_ESBM_DIR + "/in_embed/label_{}_vec.npz".format(model_name), vec_ndarray
        )
        # FACES
    elif dataset == "faces":
        with open(IN_ESBM_DIR + "/dsFACES_filtered/elist.txt") as f:
            for line in f:
                if index is 0:
                    index += 1
                    continue
                item = line.strip().split("\t")
                eid = int(item[0])
                elabel = item[3]
                eid_label_dict[eid] = elabel
        # convert entity label to word vec , entity index : label_vec, average vec_matrix: scale the matrix to [-1,1]
        vec_matrix = []
        for item in eid_label_dict.values():
            words_vec = get_avg_vec(item)
            vec_matrix.append(words_vec)
        vec_ndarray = np.array(vec_matrix)
        print("label vec shape is {}".format(vec_ndarray.shape[1]))
        np.savez(
            IN_ESBM_DIR + "/in_embed/{}_label_{}_vec.npz".format("faces", model_name),
            vec_ndarray,
        )
    else:
        print("can not find dataset {}".format(dataset))


def get_avg_vec(
    input_word,
    split_limiter=" ",
):
    # using fasttext to generate word vectors
    words = input_word.split(split_limiter)
    if len(words) == 1:
        if model_name == "fasttext":
            words_vec = model.get_word_vector(input_word)
            words_vec = np.array(words_vec)
        else:
            inputs = tokenizer(words, return_tensors="pt")["input_ids"]
            output = embed(inputs)
            pooled_output = output.squeeze(0).mean(0)
            words_vec = pooled_output.detach().numpy()

    else:
        words_vec = 0
        for word in words:
            if word is "":
                word = "UNK"
            if model_name == "fasttext":
                word_vec = model.get_word_vector(word)
                words_vec += word_vec
            else:
                inputs = tokenizer(word, return_tensors="pt")["input_ids"]
                output = embed(inputs)
                pooled_output = output.squeeze(0).mean(0)
                word_vec = pooled_output.detach().numpy()
                words_vec += word_vec
        words_vec = np.array(words_vec / len(words))
    return words_vec


def get_pv_vector(dataset):
    property_frequency = {}
    tids_file_path = path.join(IN_ESBM_DIR, "in_embed", "{}_tids.txt".format(dataset))
    eids = []
    with open(tids_file_path) as f:
        for line in f:
            item = line.rstrip().split("\t")
            eids.append(item[0])
    db_path = os.path.join(
        IN_ESBM_DIR, "in_ESBM_benchmark_v1.2", "{}_data".format(dataset)
    )
    eid2did_dict = get_eid2did_dict(dataset)
    desc_features_matrix = {}
    for entity in eids:
        temp_pf = {}
        desc_id_list = eid2did_dict[int(entity)]
        triples = prepare_data(db_path, entity)
        index = 0
        for triple in triples:
            desc_id = desc_id_list[index]
            property = triple[5]
            if property not in temp_pf.keys():
                temp_pf[property] = []
                temp_pf[property].append(desc_id)
            else:
                temp_pf[property].append(desc_id)
            value = triple[6]
            property_vec = get_avg_vec(property)
            value_vec = get_avg_vec(value, split_limiter="_")
            pv_vec = np.concatenate([value_vec, property_vec], axis=-1)
            pv_vec = list(pv_vec)
            desc_features_matrix[desc_id] = pv_vec
            index += 1
        property_frequency[entity] = temp_pf
    print("desc vec shape is {}".format(len(desc_features_matrix[desc_id_list[0]])))
    np.savez(
        IN_ESBM_DIR + "/in_embed/{}_{}_vec.npz".format(dataset, model_name),
        pvembedding_ftaw=desc_features_matrix,
    )
    map_file = IN_ESBM_DIR + "/in_embed/property_frequency_{}.json".format(dataset)
    with open(map_file, "w") as f:
        json.dump(property_frequency, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--model_name", type=str, default="clip-vit-base-patch32")
    args_parser.add_argument("--dataset", type=str, default="ESBM")
    args = args_parser.parse_args()
    model_name = args.model_name
    if model_name == "clip-vit-base-patch32":
        model = CLIPTextModel.from_pretrained("openai/" + model_name)
        embed = model.get_input_embeddings()
        tokenizer = AutoTokenizer.from_pretrained("openai/" + model_name)
    elif model_name == "bert-base-uncased":
        model = BertModel.from_pretrained(model_name)
        embed = model.get_input_embeddings()
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name == "bert-large-uncased":
        model = BertModel.from_pretrained(model_name)
        embed = model.get_input_embeddings()
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name == "fasttext":
        model = fasttext.load_model(IN_ESBM_DIR + "/wiki.en.bin")
    else:
        print("model do not exist")
        exit(0)
    get_label_vector(args.dataset)
    if args.dataset == "ESBM":
        for dataset in ["dbpedia", "lmdb"]:
            get_pv_vector(dataset)
    else:
        get_pv_vector(args.dataset)
    print("finish generating vec")
