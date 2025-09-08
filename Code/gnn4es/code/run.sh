python code/train_test.py \
--dataset lmdb \
--alpha 0 \
--lr 0.001 \
--epochs 50 \
--backbone DisenGCN \
--device cuda:1 \
--valid_epoch 1 \
--topk 10 \
--sub_graph_num 4 \
--triplet_margin 0.1 \
--dcl_output_layers 4 \
--pre_trained_model "clip-vit-base-patch32" \
--out_to_files true \
--neg_sample true \

# --pre_trained_model "bert-large-uncased" \
# --pre_trained_model "bert-base-uncased" \
# --pre_trained_model "fasttext" \