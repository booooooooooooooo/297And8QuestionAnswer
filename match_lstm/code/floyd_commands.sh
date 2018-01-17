#!/bin/bash
#usage: train, valid and test model


#Must be consistent with preprocess
pass_max_length=766
ques_max_length=60
batch_size=97
embed_size=50

######train#######
##################
num_units=10
dropout=0.5
do_clip=false
clip_norm=5
optimizer="adam"
lr=10
n_epoch=5
batches_file="/data/data_feed_ready/train.batches"
dir_to_save_graph="/output/tf_graph/"
graph_path_list_json_file="/output/graph_path_list.json"
floyd run --env tensorflow-1.4:py2 --data bo.nov29/datasets/squad/1:/data "python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $batches_file $dir_to_save_graph $graph_path_list_json_file"


######valid#######
##################
valid_json="/data/data_json/valid.json"
valid_passage="/data/data_token/valid.passage"
valid_question_id="/data/data_token/valid.question_id"
valid_batches="/data/data_feed_ready/valid.batches"
best_graph_info_json_file="/output/best_graph_info.json"
floyd run --env tensorflow-1.4:py2 --data bo.nov29/datasets/squad/1:/data --data trainoutput:/output "python valid.py $valid_json $valid_passage $valid_question_id $valid_batches $graph_path_list_json_file $best_graph_info_json_file"




######test#######
##################
test_score_info_file="/output/test_score_info"
python test.py "/data/data_json/test.json" "/data/data_token/test.passage" "/data/data_token/test.question_id" "/data/data_feed_ready/test.batches" $best_graph_info_json_file $test_score_info_file
