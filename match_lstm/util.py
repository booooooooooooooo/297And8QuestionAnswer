import tensorflow as tf

def get_batches(batches_file):
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)
    return batches
