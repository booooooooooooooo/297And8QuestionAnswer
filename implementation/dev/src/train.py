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


def train(pass_max_length, ques_max_length, batch_size, embed_size, num_units, dropout, do_clip, clip_norm, optimizer, lr, n_epoch, train_batches_sub_path, dir_data, dir_output):
    model = MatchLSTMAnsPtr(pass_max_length, ques_max_length, batch_size, embed_size, num_units, dropout)
    graph_sub_paths_list = fit(model, do_clip, clip_norm, optimizer, lr, n_epoch, train_batches_sub_path, dir_data, dir_output)
    return graph_sub_paths_list

    # with open(os.path.join(dir_output, "graph_files_list.json"), 'w') as f:
    #     f.write(json.dumps(graph_files_list))
