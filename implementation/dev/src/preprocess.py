import nltk
import json
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import pickle
import random
import matplotlib.pyplot as plt

'''
nltk.word_tokenize has some weird behaviours.
First, it tokenizes ''word'' to "word". This case is not corrected in this code.
Second, it tokenizes "word" to ``word''. This caseis corrected in this code.
'''

class Preprocessor:
    def __init__(self):
        self._PAD = b"<pad>"
        self._SOS = b"<sos>"#no _SOS in this project
        self._UNK = b"<unk>"
        self._START_VOCAB = [self._PAD, self._SOS, self._UNK]

        self.PAD_ID = 0
        self.SOS_ID = 1
        self.UNK_ID = 2
    '''
    Download raw data
    '''
    def download(self):
        #TODO
        return
    '''
    Get statistics
    '''
    def analyze(self, json_file):
        #TODO: get nice statistics about data
        count = 0
        pass_max_length = 0
        pass_ave_length = 0
        ques_max_length = 0
        ques_ave_length = 0

        pass_lengths = []
        ques_lengths = []

        with open(json_file) as fh:
            data_json = json.load(fh)
        for article_id in tqdm(xrange(len(data_json['data'])), desc="Analyzing {}".format(json_file)):
            paragraphs = data_json['data'][article_id]['paragraphs']
            for paragraph_id in xrange(len(paragraphs)):
                context = paragraphs[paragraph_id]['context']
                context_token = self.tokenize(context)
                qas = paragraphs[paragraph_id]['qas']
                for qas_id in range(len(qas)):
                    question = qas[qas_id]['question']
                    question_token = self.tokenize(question)
                    answers = qas[qas_id]['answers']
                    for answer_id in xrange(len(answers)):
                        text = answers[answer_id]['text']
                        pass_lengths.append(len(context_token))
                        ques_lengths.append(len(question_token))
                        count += 1
                        pass_max_length = max(pass_max_length, len(context_token))
                        pass_ave_length += len(context_token)
                        ques_max_length = max(ques_max_length, len(question_token))
                        ques_ave_length += len(question_token)
        pass_ave_length /= count
        ques_ave_length /= count
        print "Statistics of {}".format(json_file)
        print "How many (passage, question, answer) tuples : {} \npass_max_length: {} \npass_ave_length: {}\nques_max_length: {} \nques_ave_length :{} \n".format(count, pass_max_length, pass_ave_length, ques_max_length, ques_ave_length)
        plt.figure("Passage lengths")
        plt.plot(pass_lengths)
        plt.figure("Question lengths")
        plt.plot(ques_lengths)



    '''
    Split raw data to train, valid and test
    '''
    def get_all_json(self, squad_train_json_file, squad_dev_json_file, train_percent, dir_to_save):
        if os.path.isfile(os.path.join(dir_to_save,"train.json" )):
            print "All json files are ready!"
            return

        train_percent = float(train_percent)
        with open(squad_train_json_file) as fh:
            data_json = json.load(fh)
        data = data_json['data']
        train_json = {}
        train_json['version'] = data_json['version']
        train_json['data'] = data[0: int(len(data) * train_percent)]
        with open(os.path.join(dir_to_save,"train.json" ), 'w') as f:
            f.write(json.dumps(train_json))
        valid_json = {}
        valid_json['version'] = data_json['version']
        valid_json['data'] = data[int(len(data) * train_percent) : len(data)]
        with open(os.path.join(dir_to_save,"valid.json" ), 'w') as f:
            f.write(json.dumps(valid_json))

        with open(squad_dev_json_file) as fh:
            data_json = json.load(fh)
        with open(os.path.join(dir_to_save,"test.json" ), 'w') as f:
            f.write(json.dumps(data_json))



    '''
    From json to tokens
    '''
    def tokenize(self, s):
        '''
        paras
            s: unicode
        return
            s_token: list of unicode
        '''
        s_token = nltk.word_tokenize(s)
        s_token = [token.replace("``", '"').replace("''", '"') for token in s_token]#nltk makes "aaa "word" bbb" to 'aaa', '``', 'word', '''', 'bbb'
        return s_token

    def c_id_to_token_id(self, context, context_token):
        '''
        paras
            context: unicode
            context_token: list of unicode
        '''
        c_id_to_token_id_map = {}
        token_id = 0
        id_in_cur_token = 0
        for c_id, c in enumerate(context):
            if nltk.word_tokenize(c) != []:
                try:
                    if id_in_cur_token == len(context_token[token_id]):
                        token_id += 1
                        id_in_cur_token = 0
                    c_id_to_token_id_map[c_id] = token_id
                except IndexError as e:
                    c_id_to_token_id_map[c_id] = token_id - 1#make c_id_to_token_id_map a wrong but working map
                    print context.encode('utf8')
                    print len(context)
                    print c_id
                    print c.encode('utf8')
                    print len(context_token)
                    print token_id
                    print id_in_cur_token
                    print "======================="
                id_in_cur_token += 1

        return c_id_to_token_id_map

    def get_tokens_for_train(self, json_file, dir_to_save, prefix):
        if os.path.isfile(os.path.join(dir_to_save, prefix + ".passage" )):
            print "All {} tokens are ready!".format(prefix)
            return
        passage_list = []
        question_list = []
        answer_text_list = []
        answer_span_list = []
        with open(json_file) as fh:
            data_json = json.load(fh)
        for article_id in tqdm(xrange(len(data_json['data'])), desc="Preprocessing {}".format(json_file)):
            paragraphs = data_json['data'][article_id]['paragraphs']
            for paragraph_id in xrange(len(paragraphs)):
                context = paragraphs[paragraph_id]['context']
                context_token = self.tokenize(context)
                c_id_to_token_id_map = self.c_id_to_token_id(context, context_token)
                qas = paragraphs[paragraph_id]['qas']
                for qas_id in range(len(qas)):
                    question = qas[qas_id]['question']
                    question_token = self.tokenize(question)
                    answers = qas[qas_id]['answers']
                    for answer_id in xrange(len(answers)):
                        text = answers[answer_id]['text']
                        answer_start = answers[answer_id]['answer_start']
                        answer_end = answer_start + len(text) - 1
                        a_s = c_id_to_token_id_map[answer_start]
                        a_e = c_id_to_token_id_map[answer_end]


                        passage_list.append(context_token)
                        question_list.append(question_token)
                        answer_text_list.append(text)#untokenized
                        answer_span_list.append([str(a_s), str(a_e)])

        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)

        with open(os.path.join(dir_to_save, prefix + '.passage'), 'w') as passage_tokens_file, \
             open(os.path.join(dir_to_save, prefix + '.question'), 'w') as question_tokens_file, \
             open(os.path.join(dir_to_save, prefix + '.answer_text'), 'w') as ans_text_file, \
             open(os.path.join(dir_to_save, prefix + '.answer_span'), 'w') as ans_span_file:
             for i in tqdm(xrange(len(passage_list)), desc="Writing {} tokens to {}".format(prefix, dir_to_save)):
                 passage_tokens_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                 question_tokens_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                 ans_text_file.write(answer_text_list[i].encode('utf8') + '\n')
                 ans_span_file.write(' '.join(answer_span_list[i]) + '\n')


    def get_token_for_valid_and_test(self, json_file, dir_to_save, prefix):
        #TODO: trancate passage and question here instead of in embedding?
        if os.path.isfile(os.path.join(dir_to_save, prefix + ".passage" )):
            print "All {} tokens are ready!".format(prefix)
            return

        passage_list = []
        question_list = []
        question_id_list = []
        with open(json_file) as fh:
            data_json = json.load(fh)
        for article_id in tqdm(xrange(len(data_json['data'])), desc="Preprocessing {}".format(json_file)):
            paragraphs = data_json['data'][article_id]['paragraphs']
            for paragraph_id in xrange(len(paragraphs)):
                context = paragraphs[paragraph_id]['context']
                context_token = self.tokenize(context)
                qas = paragraphs[paragraph_id]['qas']
                for qas_id in range(len(qas)):
                    question_id = qas[qas_id]['id']
                    question = qas[qas_id]['question']
                    question_token = self.tokenize(question)

                    question_id_list.append(question_id)
                    passage_list.append(context_token)
                    question_list.append(question_token)

        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)

        with open(os.path.join(dir_to_save, prefix + '.passage'), 'w') as passage_tokens_file, \
             open(os.path.join(dir_to_save, prefix + '.question'), 'w') as question_tokens_file, \
             open(os.path.join(dir_to_save, prefix + '.question_id'), 'w') as question_id_file:
             for i in tqdm(xrange(len(passage_list)), desc="Writing my {} tokens to {} folder".format(prefix, dir_to_save)):
                 passage_tokens_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                 question_tokens_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                 question_id_file.write(question_id_list[i].encode('utf8') + '\n')


    '''
    Make voc, rev_voc using train and valid tokens  (!!DO NOT use test tokens)
    '''
    def make_voc(self, voc_path , token_paths):
        voc = Set()
        for path in token_paths:
            with open(path) as f:
                for line in f:
                    for token in line.split():
                        voc.add(token.lower())
        voc = self._START_VOCAB + list(voc)
        with open(voc_path, "w") as f:
            for token in voc:
                f.write(token.encode('utf8') + "\n".encode('utf8'))



    '''
    Make embedding matrix using train and valid tokens (!!DO NOT use test tokens)
    '''
    def make_embed(self, voc_path, glove_file, embed_size, embed_path):
        #check whether glove_file contains the word vectors with embed_size
        with open(glove_file) as fh:
            line = fh.readline()
            line_list = line.split()
            if len(line_list) - 1 != embed_size:
                raise ValueError("No gloVe word vector has size {}".formate(embed_size))
        #load voc
        voc = []
        with open(voc_path) as f:
            for line in f:
                tokens = line.split()
                voc.append(tokens[0].decode('utf8'))
        #get embed_matrix
        embed_matrix = np.random.randn(len(voc), embed_size)
        with open(glove_file) as fh:
            for line in fh:
                line_list = line.split()
                word = line_list[0].decode('utf8')#decode
                vector = line_list[1:]
                if word.lower() in voc:
                    idx = voc.index(word.lower())
                    embed_matrix[idx, :] = vector
        #save embed_matrix
        np.save(embed_path, embed_matrix)
        print "Embed matrix is save in {}".format(embed_path)

    '''
    Make token_ids using tokens and vocabulary
    '''
    def tokens_to_token_ids(self, voc_file, token_file, token_id_file):
        #load voc
        voc = []
        with open(voc_path) as f:
            for line in f:
                tokens = line.split()
                voc.append(tokens[0].decode('utf8'))
        with open(token_file) as tf, open(token_id_file, 'w') as idf:
            for line in f:
                token_list = line.decode('utf8').split()
                token_id_list = [voc.index(token) for token in token_list]
                idf.write(' '.join(token_id_list) + '\n')


if __name__ == "__main__":
    my_preprocessor = Preprocessor()

    # parser = argparse.ArgumentParser(
    #     description='Preprocessing json to tokens')
    # parser.add_argument('function_name')
    # parser.add_argument('parameters', metavar='para', type=str, nargs='+',
    #                     help='sequence of parameters')
    # args = parser.parse_args()
    # getattr(my_preprocessor, args.function_name)(*args.parameters)


    '''Download raw data'''

    '''Get statistics'''
    my_preprocessor.analyze("../data/data_raw/train-v1.1.json")
    my_preprocessor.analyze("../data/data_raw/dev-v1.1.json")

    # '''Split raw data to train.json, valid.json, test.json '''
    # my_preprocessor.get_all_json("../data/data_raw/train-v1.1.json",
    #                              "../data/data_raw/dev-v1.1.json", 0.9,
    #                              "../data/data_clean/")
    #
    # '''Tokenize train.json, valid.json and test.json'''
    # my_preprocessor.get_token_with_answers("../data/data_clean/train.json" ,
    #                                        "../data/data_clean/", "train")
    # my_preprocessor.get_token_without_answers("../data/data_clean/valid.json" ,
    #                                           "../data/data_clean/", "valid")
    # my_preprocessor.get_token_without_answers("../data/data_clean/test.json" ,
    #                                           "../data/data_clean/", "test")
    #
    # '''Make voc, rev_voc using train and valid tokens  (!!DO NOT use test tokens)'''
    # my_preprocessor.make_voc("../data/data_clean/vocabulary" ,
    #                         ["../data/data_clean/train.passage", "../data/data_clean/train.question",
    #                         "../data/data_clean/valid.passage", "../data/data_clean/valid.question"])
    #
    # '''Make embedding matrix using vocabulary and glove'''
    # my_preprocessor.make_embed("../data/data_clean/vocabulary", "../data/data_raw/glove.6B/glove.6B.50d.txt", 50, "../data/data_clean/embed_matrix")
    #
    # '''Make train ids, valid ids and test ids'''
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/train.passage", "../data/data_clean/train.passage.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/train.question", "../data/data_clean/train.question.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/valid.passage", "../data/data_clean/valid.passage.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/valid.question", "../data/data_clean/valid.question.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/test.passage", "../data/data_clean/test.passage.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/test.question", "../data/data_clean/test.question.token_id")
