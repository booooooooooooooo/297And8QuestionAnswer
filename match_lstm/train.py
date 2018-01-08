import tensorflow as tf
import numpy as np
import pickle
import argparse

from model import Model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trainning tensorflow graph')
    parser.add_argument('pass_max_length')
    parser.add_argument('ques_max_length')
    parser.add_argument('batch_size')
    parser.add_argument('embed_size')
    parser.add_argument('num_units')
    parser.add_argument('optimizer')
    parser.add_argument('lr')
    parser.add_argument('n_epoch')
    parser.add_argument('batches_file')
    parser.add_argument('dirToSaveModel')
    args = parser.parse_args()

    myModel = Model(int (args.pass_max_length), int (args.ques_max_length), int (args.batch_size), int (args.embed_size), int (args.num_units))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saved_model_list = myModel.fit(sess, args.optimizer, float (args.lr), int(args.n_epoch), args.batches_file, args.dirToSaveModel)
