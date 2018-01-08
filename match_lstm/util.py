import tensorflow as tf
import pickle

def get_batches(batches_file):
    print "Reading batched data from disk......."
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)
    return batches
