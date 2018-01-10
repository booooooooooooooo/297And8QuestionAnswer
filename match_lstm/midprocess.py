from tqdm import tqdm
import pickle
import os
import argparse

'''
Convert token to vector
'''
class Midprocessor:
    def __init__(self, pass_max_length, ques_max_length, batch_size, embed_size):
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length
        self.batch_size = batch_size
        self.embed_size = embed_size
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
            for line in tqdm(fh, desc="Reading {}".format(glove_path)):
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
        passage_sequence_length = []
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
                passage_sequence_length.append(min(len(token_list), pass_max_length))
        #pad or strip each question to same ques_max_length, vectorization
        question_vectors = []
        question_sequence_length = []
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
                question_sequence_length.append(min(len(token_list), ques_max_length))
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
        passage_sequence_length += [pass_max_length] * amount
        question_vectors += [question_fake for i in xrange(amount)]
        question_sequence_length += [ques_max_length] * amount
        answer_spans += [answer_span_fake for i in xrange(amount)]

        batches = []
        for i in tqdm(range(0, len(passage_vectors) , batch_size), desc="Split vectors to batches") :
            batch_passage = passage_vectors[i: i + batch_size]
            batch_passage_sequence_length = passage_sequence_length[i: i + batch_size]
            batch_question = question_vectors[i: i + batch_size]
            batch_question_sequence_length = question_sequence_length[i: i + batch_size]
            batch_answer_span = answer_spans[i: i + batch_size]
            batches.append((batch_passage, batch_passage_sequence_length, batch_question, batch_question_sequence_length, batch_answer_span))

        return batches
    def get_padded_vectorized_and_batched(self, passage_file, question_file, answer_span_file, glove_path, batches_file_path ):
        if os.path.isfile(batches_file_path):
            print "\"{}\" already exists.".format(batches_file_path)
            return

        vocabulary = self.get_vocabulary(passage_file, question_file)
        small_glove_dic = self.get_small_size_glove(vocabulary, glove_path)
        batches = self.get_batched_vectors(passage_file, question_file, answer_span_file, small_glove_dic)
        if not os.path.isdir(os.path.dirname(batches_file_path)):
            os.makedirs(os.path.dirname(batches_file_path) )
        with open(batches_file_path, 'w') as f:
            pickle.dump(batches, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Midprocessing token to vector')
    parser.add_argument('pass_max_length')
    parser.add_argument('ques_max_length')
    parser.add_argument('batch_size')
    parser.add_argument('embed_size')
    parser.add_argument('passage_file')
    parser.add_argument('question_file')
    parser.add_argument('answer_span_file')
    parser.add_argument('glove_path')
    parser.add_argument('batches_file_path')
    parser.add_argument("function_name")
    args = parser.parse_args()

    my_midprocessor = Midprocessor(int(args.pass_max_length) , int(args.ques_max_length), int(args.batch_size), int(args.embed_size) )
    if args.function_name == "get_padded_vectorized_and_batched":
        my_midprocessor.get_padded_vectorized_and_batched(args.passage_file, args.question_file, args.answer_span_file, args.glove_path, args.batches_file_path)
