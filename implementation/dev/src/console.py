
import argparse
import pickle
import tensorflow as tf
import os

from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running match_lstm_ans_ptr')
    parser.add_argument('dir_data')
    parser.add_argument('dir_output')

    parser.add_argument('embed_matrix_file')
    parser.add_argument('pass_max_len')
    parser.add_argument('ques_max_len')
    parser.add_argument('embed_size')
    parser.add_argument('num_units')
    parser.add_argument('clip_norm')
    parser.add_argument('optimizer')
    parser.add_argument('lr')
    parser.add_argument('n_epoch')
    parser.add_argument('batch_size')
    parser.add_argument('keep_prob')

    args = parser.parse_args()
    
    dir_data = args.dir_data
    dir_output = args.dir_output
    embed_matrix_file = args.embed_matrix_file
    pass_max_len = int(args.pass_max_len)
    ques_max_len = int(args.ques_max_len)
    embed_size = int(args.embed_size)
    num_units = int(args.num_units)
    clip_norm = float(args.clip_norm)
    optimizer = args.optimizer
    lr = float(args.lr)
    n_epoch = int(args.n_epoch)
    batch_size = int(args.batch_size)
    keep_prob = float(args.keep_prob)


    stats = train(dir_data,dir_output,  embed_matrix_file, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch, batch_size, keep_prob)
