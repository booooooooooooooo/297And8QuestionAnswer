import numpy as np

class MidProcess:
    def __init__(self):
        self.pass_max_length = 766
        self.ques_max_length = 60
        self.batch_size = 97
        self.embed_size = 50
    def get_lowercase_vocabulary(self, passage_file, question_file):

    def get_small_size_glove(self, vocabulary, glove_path):

    def get_vectors(self, passage_file, question_file, glove_dic):

    def get_batchs(self, passage_vectors, question_vectors):
