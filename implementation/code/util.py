import tensorflow as tf
import pickle
import json
import numpy as np

def get_batches(batches_file, small_size = False):
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)
    if not small_size:
        return batches
    else:
        return batches[0 : len(batches) / 200]
