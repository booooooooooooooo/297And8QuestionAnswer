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

    #read train batches from dir_data
    datautil = DataUtil()
    batches = datautil.getTrainBatches(dir_data, pass_max_len, ques_max_len, batch_size, keep_prob)


    #read small_validation_dataset
    passage, passage_sequence_length, ques, ques_sequence_length, ans, passage_text, ques_text, ans_texts, voc = datautil.getValid(dir_data, pass_max_len, ques_max_len )
    indices = random.sample(range(0, len(passage)), 50)
    small_validation_dataset = ([passage[i] for i in indices],
                                [passage_sequence_length[i] for i in indices],
                                [ques[i] for i in indices],
                                [ques_sequence_length[i] for i in indices],
                                [ans[i] for i in indices],
                                [passage_text[i] for i in indices],
                                [ques_text[i] for i in indices],
                                [ans_texts[i] for i in indices],
                                voc)

    #create graph and fit
    model = MatchLSTMAnsPtr(embed_matrix, pass_max_len, ques_max_len, embed_size, num_units, clip_norm, optimizer, lr, n_epoch)
    with tf.Session() as sess:
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"
        stats = model.fit(sess, batches, small_validation_dataset, dir_output)

    # with open(os.path.join(dir_output, "train_stats.json"), 'w') as f:
    #     f.write(json.dumps(stats))
    return stats
