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
from datetime import datetime

from model_match_lstm_ans_ptr import MatchLSTMAnsPtr
from util_data import DataUtil


def train(embed_matrix_path, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch, dir_data, batch_size, keep_prob, dir_output):
    #read batches from dir_data
    print "Start reading batches from {}".format(dir_data)
    pass_token_id_file = os.path.join(dir_data, "train.passage.token_id")
    ans_span_file = os.path.join(dir_data, "train.answer_span")
    ques_token_id_file = os.path.join(dir_data, "train.question.token_id")
    pass_trim, passage_sequence_length, ans_span_trim, ques_trim, ques_sequence_length = DataUtil().prepare_train(pass_token_id_file, pass_max_len, ans_span_file, ques_token_id_file, ques_max_len)
    batches = []
    for i in xrange(len(pass_trim) / batch_size):
        batch_pass_trim = pass_trim[i * batch_size : (i + 1) * batch_size]
        batch_passage_sequence_length = passage_sequence_length[i * batch_size : (i + 1) * batch_size]
        batch_ques_trim = ques_trim[i * batch_size : (i + 1) * batch_size]
        batch_ques_sequence_length = ques_sequence_length[i * batch_size : (i + 1) * batch_size]
        batch_ans_span_trim = ans_span_trim[i * batch_size : (i + 1) * batch_size]
        batch_keep_prob = [keep_prob] * batch_size
        batches.append([batch_pass_trim, batch_passage_sequence_length, batch_ques_trim, batch_ques_sequence_length, batch_ans_span_trim, batch_keep_prob])
    #ignore the remain data
    print "Finish reading batches"

    #read embed_matrix
    print "Start reading embed_matrix from {}".format(embed_matrix_path)
    embed_matrix = np.load(embed_matrix_path)
    print "Finish reading embed_matrix"


    #create graph and fit
    model = MatchLSTMAnsPtr(embed_matrix, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch)
    with tf.Session() as sess:
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"
        stats = model.fit(sess, batches, dir_output)

    # with open(os.path.join(dir_output, "train_stats.json"), 'w') as f:
    #     f.write(json.dumps(stats))
    return stats
if __name__ == "__main__":
    # '''
    # Unit testing
    # '''
    embed_matrix_path = "../data/data_clean/embed_matrix.npy"
    pass_max_len = 99
    ques_max_len= 37
    embed_size = 50
    num_units = 50
    clip_norm = 5
    optimizer = "adam"
    lr = 1
    n_epoch = 1
    dir_data = "../data/data_clean"
    batch_size = 77
    keep_prob = 0.5
    dir_output = "../output"
    train(embed_matrix_path, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch, dir_data, batch_size, keep_prob, dir_output)
