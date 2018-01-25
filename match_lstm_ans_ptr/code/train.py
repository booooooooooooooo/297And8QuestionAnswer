'''
Used for Trainning
'''

import tensorflow as tf
import numpy as np
import pickle
import argparse
import json

from model import Model


def get_train_op(model, do_clip, clip_norm, optimizer, lr):
    if optimizer == "adam":
        optimizer_func = tf.train.AdamOptimizer(lr)
    elif optimizer == "sgd":
        optimizer_func = tf.train.GradientDescentOptimizer(lr)
    else:
        raise ValueError('Parameters are wrong')
    loss = model.loss
    gradients, variables = zip(*optimizer_func.compute_gradients(loss))
    if do_clip:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()
    return train_op

def run_batch(sess, model, train_op, batch):
    passage, passage_sequence_length, ques ,ques_sequence_length, ans = batch
    _ , batch_loss = sess.run([train_op, model.loss], {model.passage : passage,
                                                      model.passage_sequence_length : passage_sequence_length,
                                                      model.ques : ques,
                                                      model.ques_sequence_length : ques_sequence_length,
                                                      model.ans : ans})
    return batch_loss
def run_epoch(sess, model, train_op, batches):
    trainLoss = 0.0
    for i in xrange(len(batches)):
        trainLoss += run_batch(sess, model, train_op, batches[i])
    trainLoss /= len(batches)
    return trainLoss
def fit(model, do_clip, clip_norm, optimizer, lr, n_epoch, train_batches_sub_path, dir_data, dir_output):
    print "Start getting train_op"
    train_op = get_train_op(model, do_clip, clip_norm, optimizer, lr)
    print "Finish getting train_op"
    print "Start intializing graph"
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())#Initilizing after making train_op
    print "Finish intializing graph"
    print "Start Reading batched data from disk"
    batches = get_batches(os.path.join(dir_data, train_batches_sub_path))#call get_batches from util.py
    print "Finish Reading batched data from disk"
    graph_sub_paths_list = []
    for epoch in tqdm(xrange(n_epoch), desc = "Trainning {} epoches".format(n_epoch) ):
        trainLoss = run_epoch(sess, model, train_op, batches)
        graph_sub_path = os.path.join(dir_output, "/graphes", str(trainLoss) + "_" + str(optimizer) + "_" + str(lr)  + "_" + str(epoch) + "_" + str(datetime.now()))
        tf.train.Saver().save(sess, graph_sub_path )
        graph_sub_paths_list.append(graph_sub_path)
        print "Epoch {} trainLoss {}".format(epoch, trainLoss)
    return graph_sub_paths_list

def train(pass_max_length, ques_max_length, batch_size, embed_size, num_units, dropout, do_clip, clip_norm, optimizer, lr, n_epoch, train_batches_sub_path, dir_data, dir_output):
    model = Model(pass_max_length, ques_max_length, batch_size, embed_size, num_units, dropout)
    graph_sub_paths_list = fit(model, do_clip, clip_norm, optimizer, lr, n_epoch, train_batches_sub_path, dir_data, dir_output)
    return graph_sub_paths_list
