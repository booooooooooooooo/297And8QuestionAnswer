#!/bin/bash
#purpose: make the whole pipeline work

#define all parameters here to make train, valid and test consistent
#They are NOT correct parameters
pass_max_length=13
ques_max_length=7
batch_size=97
embed_size=50

######train#######
##################
#preprocess
function_name="preprocess_train_json_to_train_and_valid_token"
train_and_valid_json_file="./data_raw/train-v1.1.json"
dir_to_save="./data_token/"
train_percent=0.9
train_token_path_file="./output/train_token_path"
valid_token_path_file="./output/valid_token_path"
python preprocess.py $function_name $train_and_valid_json_file $dir_to_save $train_percent $train_token_path_file $valid_token_path_file
#midprocess
function_name="midprocess_train_token"
glove_path="./data_raw/glove.6B/glove.6B.50d.txt"
train_batches_path_file="./data_feed_ready/train.batches"
python midprocess.py $function_name $pass_max_length $ques_max_length $batch_size $embed_size $train_token_path_file $glove_path $train_batches_path_file
# #fit graph
# num_units=10
# dropout=0.5
# do_clip=false
# clip_norm=5
# optimizer="adam"
# lr=10
# n_epoch=5
# batches_file="./data_feed_ready/train.batches"
# dir_to_save_graph="./tf_graph/"
# graph_path_list_file="./output/graph_path_list"
# equipment="mac"
# python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $batches_file $dir_to_save_graph $graph_path_list_file $equipment


######valid#######
##################
#preprocess: already done in train part
# #midprocess
# valid_batches_file_path="./data_feed_ready/valid.batches"
# python midprocess.py $pass_max_length $ques_max_length $batch_size $embed_size $valid_token_path_file $glove_path $valid_batches_file_path
# #valid graphs in $graph_path_list_file
# best_graph_path_file="./output/best_graph_path"
# valid_equipment="mac"
# python valid.py $valid_batches_file_path $valid_answer_span_file $valid_passage_file $graph_path_list_file $best_graph_path_file $valid_equipment
