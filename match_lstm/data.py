'''
Used by model to get word embedding, get vocabulary, get mini_batches etc.
'''
import os
import json
class Data:
    def __init__(self):
        #TODO: self. embed_s, batch_s, pass_l, ques_l, embed_matrix
    def split_train_to_train_and_valid(filename):
        with open(filename) as data_file:
            data = json.load(data_file)
        print data['data'][0].keys()
        print data['data'][0]['title']
        print data['data'][0]['paragraphs'][1].keys()
