'''
Used by model to get word embedding, get vocabulary, get mini_batches etc.
'''
#TODO: evaluation mode

import os
import json

class Data:
    def __init__(self, batch_s, file_data, file_glove):
        self.embed_s = 100
        self.batch_s = batch_s
        self.file_data = file_data
        self.file_glove = file_glove

        self.addVocabulary()
        self.addTrainValidBatch()
        self.addEmbedMatrix()

    def addVocabulary(self):
        #TODO: self. pass_l, ques_l, vocabulary, vocabulary_rev
        file_data = self.file_data
        with open(file_data) as data_file:
            data = json.load(data_file)
        print data['data'][0].keys()
        print data['data'][0]['title']
        print data['data'][0]['paragraphs'][1].keys()
    def addTrainValidBatch(self):
        #TODO: self.train_batch_list, valid_batch_list
    def addEmbedMatrix(self):
        #TODO: self.embed_matrix
    
