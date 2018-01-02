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
        '''
        Use zero vector to pad
        Use zero vector if there is no glove vector
        '''
        #pad or strip each passage to same pass_max_length
        pass_max_length = self.pass_max_length
        pass_zero_vector = [0] * pass_max_length

        passage_vectors = []
        with open(passage_file) as fh:
            for line in fh:
                datum = []
                token_list = line.split()
                for i in range(0, min(len(token_list), pass_max_length)):
                    token = token_list[i]
                    word = token.lower()
                    if word in small_glove_dic:
                        datum.append(small_glove_dic[word])
                    else:
                        datum.append(pass_zero_vector)
                for i in range(len(token_list), pass_max_length):
                    datum.append(pass_zero_vector)
                passage_vectors.append(datum)
        #pad or strip each question to same ques_max_length
        ques_max_length = self.ques_max_length
        ques_zero_vector = [0] * ques_max_length

        question_vectors = []
        with open(question_file) as fh:
            for line in fh:
                datum = []
                token_list = line.split()
                for i in range(0, min(len(token_list), ques_max_length)):
                    token = token_list[i]
                    word = token.lower()
                    if word in small_glove_dic:
                        datum.append(small_glove_dic[word])
                    else:
                        datum.append(ques_zero_vector)
                for i in range(len(token_list), ques_max_length):
                    datum.append(ques_zero_vector)
                question_vectors.append(datum)
        #split to batches
        batch_size = self.batch_size
        #TODO

        return passage_vectors, question_vectors
