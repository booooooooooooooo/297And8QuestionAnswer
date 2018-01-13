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
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_with_answers" "./data_json/train.json" "./data_token/" "train"
python preprocess.py "get_batches_with_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/train.passage" "./data_token/train.question" "./data_token/train.answer_span" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "train"
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
graph_path_list_file="./output/graph_path_list"
# python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $batches_file $dir_to_save_graph $graph_path_list_file

######valid#######
##################
#preprocess
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_without_answers" "./data_json/valid.json" "./data_token/" "valid"
python preprocess.py "get_batches_without_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/valid.passage" "./data_token/valid.question" "./data_token/valid.question_id" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "valid"
#valid
best_graph_info_file="./output/best_graph_info"
# python valid.py "./data_json/valid.json" "./data_token/valid.passage" "./data_token/valid.question_id" "./data_feed_ready/valid.batches" $graph_path_list_file $best_graph_info_file


######test#######
##################
#preprocess
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_without_answers" "./data_json/test.json" "./data_token/" "test"
python preprocess.py "get_batches_without_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/test.passage" "./data_token/test.question" "./data_token/test.question_id" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "test"
#testing
test_score_info_file="./output/test_score_info"
python test.py "./data_json/test.json" "./data_token/test.passage" "./data_token/test.question_id" "./data_feed_ready/test.batches" $best_graph_info_file $test_score_info_file
