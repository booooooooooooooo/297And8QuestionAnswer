import numpy as np
from tqdm import tqdm

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
        #check whether glove_path contains the word vectors with self.embed_size
        with open(glove_path) as fh:
            line = fh.readline()
            line_list = line.split()
            if len(line_list) - 1 != self.embed_size:
                raise ValueError('Glove file does not match target embed_size')

        glove_dic = {}
        with open(glove_path) as fh:
            for line in tqdm(fh, desc="Preprocessing {}".format(glove_path)):
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

    def get_batched_vectors(self, passage_file, question_file, answer_span_file, small_glove_dic):
        '''
        Use zero vector to pad
        Use zero vector if there is no glove vector
        '''
        embed_size= self.embed_size
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        batch_size = self.batch_size

        zero_vector = [0] * embed_size

        #pad or strip each passage to same pass_max_length, vectorization
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
                        datum.append(zero_vector)
                for i in range(len(token_list), pass_max_length):
                    datum.append(zero_vector)
                passage_vectors.append(datum)
        #pad or strip each question to same ques_max_length, vectorization
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
                        datum.append(zero_vector)
                for i in range(len(token_list), ques_max_length):
                    datum.append(zero_vector)
                question_vectors.append(datum)
        #read in answer_span
        answer_spans = []
        with open(answer_span_file) as fh:
            for line in fh:
                datum = line.split()
                answer_spans.append(datum)
        #split to batches
        passage_fake = [zero_vector for i in xrange(pass_max_length)]
        question_fake = [zero_vector for i in xrange(ques_max_length)]
        answer_span_fake = [0, 0]

        amount = batch_size - len(passage_vectors) % batch_size

        passage_vectors += [passage_fake for i in xrange(amount)]
        question_vectors += [question_fake for i in xrange(amount)]
        answer_spans += [answer_span_fake for i in xrange(amount)]

        batches = []
        for i in tqdm(range(0, len(passage_vectors) , batch_size), desc="Batching") :
            batch_passage = passage_vectors[i: i + batch_size]
            batch_question = question_vectors[i: i + batch_size]
            batch_answer_span = answer_spans[i: i + batch_size]
            batches.append((batch_passage, batch_question, batch_answer_span))

        return batches, passage_vectors, question_vectors, answer_spans
    def get_padded_vectorized_and_batched(self, passage_file, question_file, answer_span_file, glove_path ):
        vocabulary = self.get_vocabulary(passage_file, question_file)
        small_glove_dic = self.get_small_size_glove(vocabulary, glove_path)
        batches, _, _, _ = self.get_batched_vectors(passage_file, question_file, answer_span_file, small_glove_dic)
        #TODO: save feed-ready batches to disk
