#!/bin/bash

DEVICE="cuda:1"
BACKBONE="DisenGCN"
PRE_TRAINED_MODEL="clip-vit-base-patch32"
# Run the first task
gnome-terminal -- sh -c "python code/train_test.py --dataset lmdb --backbone $BACKBONE --device $DEVICE --valid_epoch 1 --topk 5 --pre_trained_model $PRE_TRAINED_MODEL ; bash"

# Run the second task
gnome-terminal -- sh -c "python code/train_test.py --dataset lmdb --backbone $BACKBONE --device $DEVICE --valid_epoch 1 --topk 10 --pre_trained_model $PRE_TRAINED_MODEL ; bash"

# Run the third task
gnome-terminal -- sh -c "python code/train_test.py --dataset dbpedia --backbone $BACKBONE --device $DEVICE --valid_epoch 1 --topk 5 --pre_trained_model $PRE_TRAINED_MODEL ; bash"

# Run the fourth task
gnome-terminal -- sh -c "python code/train_test.py --dataset dbpedia --backbone $BACKBONE --device $DEVICE --valid_epoch 1 --topk 10 --pre_trained_model $PRE_TRAINED_MODEL ; bash"
