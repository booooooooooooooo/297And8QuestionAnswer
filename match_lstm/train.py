

import tensorflow as tf
import numpy as np

import pickle

def get_batches(batches_file):
    with open(batches_file, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        batches = pickle.load(f)
    return batches
