
import argparse
import pickle
import tensorflow as tf
import os

from train import train

if __name__ == "__main__":
    dir_data
    dir_output
    embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size

    train(dir_data, embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output)
