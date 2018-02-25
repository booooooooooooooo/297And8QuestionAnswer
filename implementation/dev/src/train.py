import tensorflow as tf
import numpy as np
import pickle
import argparse
import json
import os
from tqdm import tqdm
import random

from model import Model
from util_data import *


def train(dir_data, embed_size, pass_max_length, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output):
    #read embed_matrix
    embed_matrix_file = "word.vector.{}.npy".format(embed_size)
    print "Start reading embed_matrix from {}".format(embed_matrix_file)
    embed_matrix = np.load(os.path.join(dir_data, embed_matrix_file))
    embed_matrix = embed_matrix.astype(np.float32)
    # print type(embed_matrix[0][0])
    print "Finish reading embed_matrix"

    #create graph
    my_model = Model(embed_matrix, pass_max_length, ques_max_length, embed_size, num_units, clip_norm, lr, n_epoch, reg_scale)

    #read data
    train_data = get_data_tuple("train", dir_data, pass_max_length, ques_max_length)
    valid_data = get_data_tuple("valid", dir_data, pass_max_length, ques_max_length)
    test_data = get_data_tuple("test", dir_data, pass_max_length, ques_max_length)

    #train, valid and test graph
    with tf.Session() as sess:
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"
        my_model.fit(sess, train_data, valid_data, test_data, batch_size, sample_size, dir_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("machine")
    args = parser.parse_args()

    if args.machine == "local":

        dir_data = "../data/data_clean"
        dir_output = "../output"
        embed_size = 100
        pass_max_len = 9
        ques_max_length = 5
        num_units = 7
        clip_norm = 10
        lr = 2e-3
        n_epoch = 1
        reg_scale = 0.001
        batch_size = 32
        sample_size = 200

        train(dir_data, embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output)

    elif args.machine == "floyd":
        '''
        data_mount="bo.nov29/datasets/squad/5"
        floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python train.py floyd"
        floyd run --env tensorflow-1.4:py2 --data $data_mount:/data "python train.py floyd"
        '''
        dir_data="/data"
        dir_output="/output"

        embed_size = 100
        pass_max_len = 400
        ques_max_length = 30
        num_units = 64
        clip_norm = 10
        lr = 2e-3
        n_epoch = 1
        reg_scale = 0.001
        batch_size = 32
        sample_size = 200

        train(dir_data, embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output)

    else:
        print "Wrong machine!"
