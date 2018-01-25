'''
Used for Trainning.

TO RUN
python train.py para1 para2 ... paran <input-batches-file-path> <output_graph_file_name_list_path>
'''
import tensorflow as tf
import numpy as np
import pickle
import argparse
import json

from model import Model


'''
Below are functins used by train.py to train the graph
'''
def get_train_op(self, optimizer, lr):
    print "Getting {} optimizer".format(optimizer)

    loss = self.loss
    do_clip = self.do_clip
    clip_norm = self.clip_norm

    if optimizer == "adam":
        optimizer_func = tf.train.AdamOptimizer(lr)
    elif optimizer == "sgd":
        optimizer_func = tf.train.GradientDescentOptimizer(lr)
    else:
        raise ValueError('Parameters are wrong')

    gradients, variables = zip(*optimizer_func.compute_gradients(loss))
    if do_clip:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()

    print "Finish getting {} optimizer".format(optimizer)

    return train_op
def run_batch(self, sess, train_op, batch):
    passage, passage_sequence_length, ques ,ques_sequence_length, ans = batch
    _ , batch_loss = sess.run([train_op, self.loss], {self.passage : passage,
                                                      self.passage_sequence_length : passage_sequence_length,
                                                      self.ques : ques,
                                                      self.ques_sequence_length : ques_sequence_length,
                                                      self.ans : ans})
    # print batch_loss
    return batch_loss
def run_epoch(self, sess, train_op, batches):
    trainLoss = 0.0
    for i in xrange(len(batches)):
        trainLoss += self.run_batch(sess, train_op, batches[i])
    trainLoss /= len(batches)
    return trainLoss
def fit(self, sess, optimizer, lr, n_epoch, batches_file, dirToSaveModel, small_size = True):
    train_op = self.get_train_op(optimizer, lr)
    print "Initilizing all varibles"
    sess.run(tf.global_variables_initializer())#Initilizing after making train_op
    print "Finish Initilizing all varibles"
    batches = get_batches(batches_file, small_size)#call get_batches from util.py
    print "Finish Reading batched data from disk......."
    saved_model_list = []
    for epoch in tqdm(xrange(n_epoch), desc = "Trainning {} epoches".format(n_epoch) ):
        trainLoss = self.run_epoch(sess, train_op, batches)
        file_to_save_model = os.path.join(dirToSaveModel, str(trainLoss) + "_" + str(optimizer) + "_" + str(lr)  + "_" + str(epoch) + "_" + str(datetime.now()))
        tf.train.Saver().save(sess, file_to_save_model )
        saved_model_list.append(file_to_save_model)
        print "Epoch {} trainLoss {}".format(epoch, trainLoss)

    return saved_model_list
'''
Above are functins used by train.py to train the graph
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trainning tensorflow graph')
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
    parser.add_argument('batches_file')
    parser.add_argument('dir_to_save_graph')
    args = parser.parse_args()

    myModel = Model(int (args.pass_max_length), int (args.ques_max_length), int (args.batch_size), int (args.embed_size), int (args.num_units), float(args.dropout), bool(args.do_clip), float(args.clip_norm))

    sess = tf.Session()
    saved_model_list = myModel.fit(sess, args.optimizer, float (args.lr), int(args.n_epoch), args.batches_file, args.dir_to_save_graph, small_size = True)

    # print saved_model_list
