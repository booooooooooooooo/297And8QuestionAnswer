import tensorflow as tf
import pickle

def get_batches(batches_file, small_size = False):

    print "Reading batched data from disk......."
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)
    if not small_size:
        return batches
    else:
        return batches[0 : len(batches) / 200]
