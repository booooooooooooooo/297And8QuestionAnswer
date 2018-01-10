#!/bin/bash

#define all parameters here to make train, valid and test consistent
# pass_max_length=766
# ques_max_length=60
# batch_size=97
# embed_size=50
pass_max_length=13
ques_max_length=7
batch_size=97
embed_size=50

##train
#preprocess
pass_ques_ans_json_path="./data_raw/train-v1.1.json"
dir_to_save="./data_token/"
train_percent=0.9
python preprocess.py $pass_ques_ans_json_path $dir_to_save $train_percent
#midprocess
passage_file="./data_token/train.passage"
question_file="./data_token/train.question"
answer_span_file="./data_token/train.answer_span"
glove_path="./data_raw/glove.6B/glove.6B.50d.txt"
batches_file_path="./data_feed_ready/train.batches"
function_name="get_padded_vectorized_and_batched"
python midprocess.py $pass_max_length $ques_max_length $batch_size $embed_size $passage_file $question_file $answer_span_file $glove_path $batches_file_path $function_name
#fit graph
num_units=10
dropout=0.5
do_clip=false
clip_norm=5
optimizer="adam"
lr=10
n_epoch=5
batches_file="./data_feed_ready/train.batches"
dir_to_save_graph="./tf_graph/"
file_to_save_graph_name_list="./output/graph_name_list"
equipment="mac"
python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $batches_file $dir_to_save_graph $file_to_save_graph_name_list $equipment
