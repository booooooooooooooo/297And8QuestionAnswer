'''
TO run on floyd:

data_mount="bo.nov29/datasets/squad/2"
datafolder = "/data"
outputfolder="/output"
pass_max_length=766
ques_max_length=60
batch_size=97
embed_size=50

floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py ............"



To run on mac:


dir_data = "../mac/data"
dir_output="../mac/output"
pass_max_length=13
ques_max_length=7
batch_size=97
embed_size=50

num_units=10
dropout=0.5
do_clip=false
clip_norm=5
optimizer="adam"
lr=10
n_epoch=5
train_batches_sub_path="/data_feed_ready/train.batches"
valid_json_sub_path="/data_json/valid.json"
valid_passage_tokens_sub_path="/data_token/valid.passage"
valid_question_ids_sub_path="/data_token/valid.question_id"
valid_batches_sub_path="/data_feed_ready/valid.batches"
test_json_sub_path="/data_json/test.json"
test_passage_tokens_sub_path="/data_token/test.passage"
test_question_ids_sub_path="/data_token/test.question_id"
test_batches_sub_path="/data_feed_ready/test.batches"

python console.py $dir_data $dir_output $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $train_batches_sub_path $valid_json_sub_path $valid_passage_tokens_sub_path $valid_question_ids_sub_path $valid_batches_sub_path $test_json_sub_path $test_passage_tokens_sub_path $test_question_ids_sub_path $test_batches_sub_path


'''


import argparse
import pickle
import tensorflow as tf
import os

from train import train
from valid import valid
from test import test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running match_lstm_ans_ptr')
    parser.add_argument('dir_data')
    parser.add_argument('dir_output')

    parser.add_argument('pass_max_length')
    parser.add_argument('ques_max_length')
    parser.add_argument('batch_size')
    parser.add_argument('embed_size')
    parser.add_argument('num_units')
    parser.add_argument('dropout')
    parser.add_argument('do_clip')
    parser.add_argument('clip_norm')
    parser.add_argument('optimizer')
    parser.add_argument('lr')
    parser.add_argument('n_epoch')
    parser.add_argument('train_batches_sub_path')

    parser.add_argument('valid_json_sub_path')
    parser.add_argument('valid_passage_tokens_sub_path')
    parser.add_argument('valid_question_ids_sub_path')
    parser.add_argument('valid_batches_sub_path')

    parser.add_argument('test_json_sub_path')
    parser.add_argument('test_passage_tokens_sub_path')
    parser.add_argument('test_question_ids_sub_path')
    parser.add_argument('test_batches_sub_path')

    args = parser.parse_args()


    graph_sub_paths_list = train(int (args.pass_max_length), int (args.ques_max_length), int (args.batch_size), int (args.embed_size), int (args.num_units), float(args.dropout), bool(args.do_clip), float(args.clip_norm), args.optimizer, float (args.lr), int(args.n_epoch), args.train_batches_sub_path, args.dir_data, args.dir_output)

    valid_result = valid(args.valid_json_sub_path, args.valid_passage_tokens_sub_path, args.valid_question_ids_sub_path, args.valid_batches_sub_path, graph_sub_paths_list , args.dir_data, args.dir_output)

    test_result = test(args.test_json_sub_path, args.test_passage_tokens_sub_path, args.test_question_ids_sub_path, args.test_batches_sub_path, valid_result, args.dir_data, args.dir_output)

    print valid_result
    print test_result
