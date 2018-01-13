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
    parser.add_argument("graph_path_list_file")
    args = parser.parse_args()

    myModel = Model(int (args.pass_max_length), int (args.ques_max_length), int (args.batch_size), int (args.embed_size), int (args.num_units), float(args.dropout), bool(args.do_clip), float(args.clip_norm))

    sess = tf.Session()
    saved_model_list = myModel.fit(sess, args.optimizer, float (args.lr), int(args.n_epoch), args.batches_file, args.dir_to_save_graph, small_size = True)

    with open(args.graph_path_list_file, 'w') as f:
        f.write(json.dumps(saved_model_list))
