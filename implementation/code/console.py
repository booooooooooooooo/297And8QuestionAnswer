'''
as the file name

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
