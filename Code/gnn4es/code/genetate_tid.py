# generate eid 2 did / eid 2 smset of faces

import os

IN_DATA_DIR = os.path.join(os.getcwd(), "data")


def get_eid2did(dataset):
    if dataset == "faces":
        tid_index = 1
        eid2did_map_dict = {}
        desc2index_dict = {}
        data_dir = os.path.join(IN_DATA_DIR, "dsFACES_filtered", "dsfaces_data")
        for i in range(1, 51):
            dir = os.path.join(data_dir, "{}".format(i))
            with open(os.path.join(dir, "{}_desc.nt".format(i))) as f:
                tid_list = []
                for line in f:
                    tid_list.append(tid_index)
                    desc2index_dict[line] = tid_index
                    tid_index += 1
            eid2did_map_dict[i] = tid_list
        return eid2did_map_dict, desc2index_dict


def geteid2sid(desc2index_dict, topk):
    data_dir = os.path.join(IN_DATA_DIR, "dsFACES_filtered", "dsfaces_data")
    eid2sid_map_dict = {}
    for i in range(1, 51):
        dir = os.path.join(data_dir, "{}".format(i))
        sid_list = []
        for file in os.listdir(dir):
            # with open(os.path.join(dir, '{}_gold_top{}_{}.nt'.format(i,str(topk),j))) as f:
            if file.startswith("{}_gold_top{}_".format(i, str(topk))) and file.endswith(
                ".nt"
            ):
                f = open(os.path.join(dir, file))
                sids = []
                for line in f:
                    if line in desc2index_dict.keys():
                        sids.append(desc2index_dict[line])
                sid_list.append(sids)
        eid2sid_map_dict[i] = sid_list
    return eid2sid_map_dict


if __name__ == "__main__":
    eid2did_map_dict, desc2index_dict = get_eid2did("faces")
    with open(os.path.join(IN_DATA_DIR, "in_embed", "faces_tids.txt"), "w") as f:
        for eid in eid2did_map_dict.keys():
            f.write("{}\t{}\n".format(eid, eid2did_map_dict[eid]))
    for topk in [5, 10]:
        eid2sid_map_dict = geteid2sid(desc2index_dict, topk)
        with open(
            os.path.join(
                IN_DATA_DIR, "in_embed", "faces_egolds_top{}.txt".format(str(topk))
            ),
            "w",
        ) as f:
            for eid in eid2sid_map_dict.keys():
                f.write("{}\t{}\n".format(eid, eid2sid_map_dict[eid]))
    print("finish generate eid2did and eid2sid of faces")
