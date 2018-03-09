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


def train(dir_data, dir_output, config):
    #read embed_matrix
    embed_matrix_file = "word.vector.{}.npy".format(config["embed_size"])
    print "Start reading embed_matrix from {}".format(embed_matrix_file)
    embed_matrix = np.load(os.path.join(dir_data, embed_matrix_file))
    embed_matrix = embed_matrix.astype(np.float32)
    # print type(embed_matrix[0][0])
    print "Finish reading embed_matrix"

    #create graph
    my_model = Model(embed_matrix, config)

    #read data
    train_data = get_data_tuple("train", dir_data, config["pass_max_length"], config["ques_max_length"])
    valid_data = get_data_tuple("valid", dir_data, config["pass_max_length"], config["ques_max_length"])
    test_data = get_data_tuple("test", dir_data, config["pass_max_length"], config["ques_max_length"])

    #train, valid and test graph
    with tf.Session() as sess:
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"
        my_model.fit(sess, train_data, valid_data, test_data, config["batch_size"],
                     config["sample_size"], dir_output)


if __name__ == "__main__":

    config_match_simple = {"embed_size":100, "pass_max_length": 400, "ques_max_length": 30,
                           "num_units": 64, "clip_norm": 10, "lr": 2e-3, "n_epoch": 1,
                           "reg_scale": 0.001, "batch_size": 32, "sample_size": 200, "arch": "match_simple"}
    config_match = {"embed_size":100, "pass_max_length": 400, "ques_max_length": 30,
                    "num_units": 64, "clip_norm": 10, "lr": 2e-3, "n_epoch": 1,
                    "reg_scale": 0.001, "batch_size": 32, "sample_size": 200, "arch": "match"}
    config_r_net = {"embed_size":100, "pass_max_length": 400, "ques_max_length": 30,
                    "num_units": 64, "clip_norm": 10, "lr": 2e-3, "n_epoch": 1,
                    "reg_scale": 0.001, "batch_size": 32, "sample_size": 200, "arch": "r_net"}
    config_r_net_iter = {"embed_size":100, "pass_max_length": 400, "ques_max_length": 30,
                         "num_units": 64, "clip_norm": 10, "lr": 2e-3, "n_epoch": 1,
                         "reg_scale": 0.001, "batch_size": 32, "sample_size": 200, "arch": "r_net_iter"}

    config = {"match_simple" : config_match_simple, "match": config_match,
              "r_net": config_r_net, "r_net_iter" : config_r_net_iter}

    parser = argparse.ArgumentParser()
    parser.add_argument('--sanity', dest='mode', action='store_const',
                        const= "sanity", default="train",
                        help='Sanity the QA system(default: Train the QA system)')
    #TODO: make machine and arch optinal
    parser.add_argument("machine")
    parser.add_argument("arch")
    args = parser.parse_args()

    #choose machine
    if args.machine == "local":
        dir_data = "../data/data_clean"
        dir_output = "../output"
    elif args.machine == "floyd":
        '''
        data_mount="bo.nov29/datasets/squad/5"
        floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python train.py floyd arch"
        floyd run --env tensorflow-1.4:py2 --data $data_mount:/data "python train.py floyd arch"
        '''
        dir_data="/data"
        dir_output="/output"
    else:
        raise ValueError('Machine should be local or floyd')

    #choose arch
    if args.arch in config:
        if args.mode == "sanity":
            config_sanity = {"embed_size": 100, "pass_max_length": 9,"ques_max_length": 5,
                             "num_units": 7, "clip_norm": 10, "lr": 2e-3, "n_epoch": 1,
                             "reg_scale": 0.001, "batch_size": 32, "sample_size": 200}
            config_sanity["arch"] = args.arch
            train(dir_data, dir_output, config_sanity)
        else:
            train(dir_data, dir_output, config[args.arch])
    else:
        raise ValueError('Architecture should be match_simple, match, r_net or r_net_iter')
