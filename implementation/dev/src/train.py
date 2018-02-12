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


def train(dir_data, dir_output, embed_matrix_file, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch, batch_size, keep_prob):
    # print type(keep_prob)


    #read embed_matrix
    print "Start reading embed_matrix from {}".format(embed_matrix_file)
    embed_matrix = np.load(os.path.join(dir_data, embed_matrix_file))
    embed_matrix = embed_matrix.astype(np.float32)
    # print type(embed_matrix[0][0])
    print "Finish reading embed_matrix"


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
        batches.append([batch_pass_trim, batch_passage_sequence_length, batch_ques_trim, batch_ques_sequence_length, batch_ans_span_trim, keep_prob])
    #ignore the remain data
    print "Finish reading batches"

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
