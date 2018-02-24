'''
Used for Trainning
'''

import tensorflow as tf
import numpy as np
import pickle
import argparse
import json
import os
from tqdm import tqdm
import random




def train(dir_data, embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output)::
    #read embed_matrix
    embed_matrix_file = "word.vector.{}.npy".format(embed_size)
    print "Start reading embed_matrix from {}".format(embed_matrix_file)
    embed_matrix = np.load(os.path.join(dir_data, embed_matrix_file))
    embed_matrix = embed_matrix.astype(np.float32)
    # print type(embed_matrix[0][0])
    print "Finish reading embed_matrix"

    #read train_data
    train_data = get_data_tuple("train", dir_data, pass_max_len, ques_max_length)

    #read valid_data
    valid_data = get_data_tuple("valid", dir_data, pass_max_len, ques_max_length)

    #create graph
    my_model = Model(embed_matrix, pass_max_length, ques_max_length, embed_size, num_units, clip_norm, lr, n_epoch, reg_scale)

    #train graph
    with tf.Session() as sess:
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"
        model.fit(sess, train_data, valid_data, batch_size, sample_size, dir_output)
