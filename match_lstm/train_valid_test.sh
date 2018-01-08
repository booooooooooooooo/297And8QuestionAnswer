#!/bin/bash

#define all parameters here to make train, valid and test consistent
pass_max_length=766
ques_max_length=60
batch_size=97
embed_size=50

num_units=10


#test shell script

#train
optimizer="adam"
lr=10
n_epoch=5
batches_file="./data_feed_ready/train.batches"
dirToSaveModel="./tf_graph/"
python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $optimizer $lr $n_epoch $batches_file $dirToSaveModel
