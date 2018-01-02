import numpy as np

class Midprocessor:
    def __init__(self):
        self.pass_max_length = 766
        self.ques_max_length = 60
        self.batch_size = 97
        self.embed_size = 50
    def get_vocabulary(self, passage_file, question_file):
        vocabulary = set()
        with open(passage_file) as fh:
            token_list = fh.read().split()
            vocabulary.update(token_list)
        with open(question_file) as fh:
            token_list = fh.read().split()
            vocabulary.update(token_list)
        vocabulary_utf8 = set()
        for token in vocabulary:
            vocabulary_utf8.add(token.decode('utf8'))#decode
        return vocabulary_utf8

    def get_small_size_glove(self, vocabulary, glove_path):


        glove_dic = {}
        with open(glove_path) as fh:
            for line in fh:
                line_list = line.split()
                word = line_list[0].decode('utf8')#decode
                vector = line_list[1:]
                glove_dic[word] = vector

        small_glove_dic = {}
        for token in vocabulary:
            word = token.lower()# glove uses lowercase key
            if word in glove_dic:
                small_glove_dic[word] = glove_dic[word]
        return small_glove_dic #lowercase key

    def get_batched_vectors(self, passage_file, question_file, small_glove_dic):
        #pad of strip to same max_length

        #split to batches
        return
