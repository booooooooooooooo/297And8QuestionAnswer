import tensorflow as tf
import pickle
import json
import numpy as np
import os

from util import *
'''
batches_file is equal or longer than token_file due to fake datum
'''
#TODO: pass sess as parameter
def get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph):
    '''
        return: json
    '''
    with tf.Session() as sess:
        #TODO: intiialzing sess is necessary? I guess no
        saver = tf.train.import_meta_graph(trained_graph + '.meta')
        saver.restore(sess, trained_graph)
        passage_ph = tf.get_default_graph().get_tensor_by_name("passage_placeholder:0")
        passage_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("passage_sequence_length_placeholder:0")
        ques_ph = tf.get_default_graph().get_tensor_by_name("question_placeholder:0")
        ques_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("question_sequence_length_placeholder:0")
        dist = tf.get_default_graph().get_tensor_by_name("dist:0")
        #get predictions_dist
        pred_dist_list = []
        batches = get_batches(batches_file)
        for batch in batches:
            passage, passage_sequence_length, ques, ques_sequence_length, _ = batch
            pred_dist = sess.run(dist, {passage_ph : passage,
                                        passage_sequence_length_ph : passage_sequence_length,
                                        ques_ph : ques,
                                        ques_sequence_length_ph : ques_sequence_length})
            pred_dist_list.append(pred_dist)
    predictions_dist = np.concatenate(pred_dist_list)#(-1, 2, pass_max_length)

    #turn predictions_dist to prediction
    pred_dic = {}
    with open(passage_tokens_file) as f1, open(question_ids_file) as f2:
        passage_tokens_list = f1.readlines()
        question_id_list = f2.readlines()

    for i in xrange(len(passage_tokens_list)):
        tokens = passage_tokens_list[i].split()
        start = np.argmax(predictions_dist[i][0])
        end = np.argmax(predictions_dist[i][1])
        answer =  ' '.join(tokens[min(start, len(tokens) - 1) : min(end + 1, len(tokens))])
        question_id = question_id_list[i].split()[0]
        pred_dic[question_id] = answer
        # print tokens
        # print answer
        # print
    return pred_dic
