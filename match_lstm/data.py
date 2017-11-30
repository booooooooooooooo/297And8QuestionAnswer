'''
Used to get word embedding, get vocabulary, get mini_batches etc.
'''

import os
import json

class Data:
    def getTrain(self, ques_ans_file, glove_file, embed_s, batch_s):
        #TODO return: pass_l, ques_l, vocabulary, vocabulary_rev, embed_matrix, batch_data
        with open(file_data) as data_file:
            data = json.load(data_file)
        print data['data'][0].keys()
        print data['data'][0]['title']
        print data['data'][0]['paragraphs'][1].keys()
    def getPredict(self, ques_file, glove_file, embed_s, batch_s):
        #TODO return: pass_l, ques_l, vocabulary, vocabulary_rev, embed_matrix, batch_data
