import nltk
import json
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import pickle
import random
from urllib import urlretrieve
import zipfile

'''
nltk.word_tokenize has some weird behaviours.

Solved:  "word" to ["``", "word", "''"]
Ignored: "''word''" to ["''word", "''"], "'word'" to ["'word","'"]. Since there is word like haven't
Error: "``" to '"' due to the solved case
'''

'''
json.load(json_file) decodes everyting to unicode automatically
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
    def download_(self, base_url, local_dir, filename, unzip = False):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        if not os.path.exists(os.path.join(local_dir, filename)):
            _ , _ = urlretrieve(base_url + filename, os.path.join(local_dir, filename))
            print "{} created!".format(os.path.join(local_dir, filename))
            if unzip:
                zip_handler = zipfile.ZipFile(os.path.join(local_dir, filename), 'r')
                zip_handler.extractall(local_dir)
                zip_handler.close()
                print "{} unziped".format(os.path.join(local_dir, filename))
        else:
            print "{} already exists!".format(os.path.join(local_dir, filename))


        return
    def download(self, local_dir):
        squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        train = "train-v1.1.json"
        dev = "dev-v1.1.json"
        glove_base_url = "http://nlp.stanford.edu/data/"
        glove_zip = "glove.6B.zip"

        self.download_(squad_base_url, local_dir, train)
        self.download_(squad_base_url, local_dir, dev)
        self.download_(glove_base_url, local_dir, glove_zip, True)



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
        map_to_token = {}
        token_id = 0
        id_in_token = 0
        for c_id, c in enumerate(context):
            if nltk.word_tokenize(c) != []:
                try:
                    if id_in_token == len(context_token[token_id]):
                        token_id += 1
                        id_in_token = 0
                    map_to_token[c_id] = token_id
                except IndexError as e:
                    map_to_token[c_id] = len(context_token) - 1#make map_to_token a wrong but working map
                    print "======================="
                    print "Fail to get answer span from context below"
                    print "======================="
                    print context.encode('utf8')
                    print "======================="
                    print "length of context: {}".format(len(context))
                    print "Fail at char id: {}, char : {}".format(c_id, c.encode('utf8'))
                    print "======================="
                    print "context tokens list is below"
                    print [token.encode('utf8') for token in context_token]
                    # print len(context_token)
                    # print token_id
                    # print id_in_token
                    print "======================="
                id_in_token += 1

        return map_to_token

    def get_tokens(self, json_file):
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
                map_to_token = self.c_id_to_token_id(context, context_token)
                qas = paragraphs[paragraph_id]['qas']
                for qas_id in range(len(qas)):
                    question = qas[qas_id]['question']
                    question_token = self.tokenize(question)
                    answers = qas[qas_id]['answers']
                    for answer_id in xrange(len(answers)):
                        text = answers[answer_id]['text']
                        answer_start = answers[answer_id]['answer_start']
                        answer_end = answer_start + len(text) - 1
                        a_s = map_to_token[answer_start]
                        a_e = map_to_token[answer_end]

                        passage_list.append(context_token)
                        question_list.append(question_token)
                        answer_text_list.append(text)#untokenized
                        answer_span_list.append([str(a_s), str(a_e)])

        return passage_list, question_list, answer_text_list, answer_span_list

    def get_tokens_unique_ques(self, json_file):
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
                    question = qas[qas_id]['question']
                    question_token = self.tokenize(question)
                    question_id = qas[qas_id]['id']

                    passage_list.append(context_token)
                    question_list.append(question_token)
                    question_id_list.append(question_id)

        return passage_list, question_list, question_id_list

    def tokenize_train(self, json_file, dir_to_save):
        if os.path.isfile(os.path.join(dir_to_save, "train.answer_span" )):
            print "All train and valid tokens are ready!"
            return

        passage_list, question_list, answer_text_list, answer_span_list = self.get_tokens(json_file)
        indices = range(len(passage_list))
        np.random.shuffle(indices)
        train_per  = 0.9


        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)
        with open(os.path.join(dir_to_save, 'train.passage'), 'w') as passage_file, \
             open(os.path.join(dir_to_save, 'train.question'), 'w') as question_file, \
             open(os.path.join(dir_to_save, 'train.answer_text'), 'w') as ans_text_file, \
             open(os.path.join(dir_to_save, 'train.answer_span'), 'w') as ans_span_file:
             for k in tqdm(xrange(int(len(indices) * train_per)), desc="Writing train tokens to {}".format(dir_to_save)):
                 i = indices[k]
                 passage_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                 question_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                 ans_text_file.write(answer_text_list[i].encode('utf8') + '\n')
                 ans_span_file.write(' '.join(answer_span_list[i]) + '\n')
        with open(os.path.join(dir_to_save, 'valid.passage'), 'w') as passage_file, \
             open(os.path.join(dir_to_save, 'valid.question'), 'w') as question_file, \
             open(os.path.join(dir_to_save, 'valid.answer_text'), 'w') as ans_text_file, \
             open(os.path.join(dir_to_save, 'valid.answer_span'), 'w') as ans_span_file:
             for k in tqdm(range(int(len(indices) * train_per), len(indices)), desc="Writing valid tokens to {}".format(dir_to_save)):
                 i = indices[k]
                 passage_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                 question_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                 ans_text_file.write(answer_text_list[i].encode('utf8') + '\n')
                 ans_span_file.write(' '.join(answer_span_list[i]) + '\n')

        print "Congs! All traina and valid tokens are created!"

    def tokenize_test(self, json_file, dir_to_save):
        if os.path.isfile(os.path.join(dir_to_save, "test.passage" )):
            print "All test tokens are ready!"
            return

        passage_list, question_list, question_id_list = self.get_tokens_unique_ques(json_file)
        indices = range(len(passage_list))
        np.random.shuffle(indices)

        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)
        with open(os.path.join(dir_to_save, 'test.passage'), 'w') as passage_file, \
             open(os.path.join(dir_to_save, 'test.question'), 'w') as question_file, \
             open(os.path.join(dir_to_save, 'test.question_id'), 'w') as question_id_file:
             for i in tqdm(indices, desc="Writing test tokens to {}".format(dir_to_save)):
                 passage_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                 question_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                 question_id_file.write(question_id_list[i].encode('utf8') + '\n')

        print "Congs! All test tokens are created"

    '''
    Make voc using train and valid tokens  (!!DO NOT use test tokens)
    '''
    def make_voc(self, voc_file , token_files):
        if os.path.isfile(voc_file):
            print "{} is ready!".format(voc_file)
            return

        if not os.path.isdir(os.path.dirname(voc_file)):
            os.makedirs(os.path.dirname(voc_file))

        voc = set()
        for token_file in token_files:
            with open(token_file) as f:
                lines = f.readlines()
                for i in tqdm(xrange(len(lines)), desc = "Making vocabulary from {}".format(token_file)):
                    line = lines[i]
                    line = line
                    for token in line.split():
                        voc.add(token)
        voc = self._START_VOCAB + sorted(list(voc))
        with open(voc_file, "w") as f:
            for token in voc:
                f.write(token + "\n")
        print "Vocabulary is save in {}".format(voc_file)


    '''
    Make embedding matrix using train and valid tokens (!!DO NOT use test tokens)
    '''
    def load_vocabulary(self, voc_file):
        voc = []
        with open(voc_file) as f:
            lines = f.readlines()
            for i in tqdm(xrange(len(lines)), desc = "Loading vocabulary form {}".format(voc_file)):
                line = lines[i]
                tokens = line.decode('utf8').split()
                voc.append(tokens[0])
        return voc
    def make_embed(self, voc_file, glove_file, embed_size, embed_path):
        if os.path.isfile(embed_path + ".npy"):
            print "{} is ready!".format(embed_path)
            return

        #check whether glove_file contains the word vectors with embed_size
        with open(glove_file) as fh:
            line = fh.readline()
            line_list = line.split()
            if len(line_list) - 1 != embed_size:
                raise ValueError("No gloVe word vector has size {}".formate(embed_size))
        #load voc
        voc = self.load_vocabulary(voc_file)
        voc_set = set(voc)
        #load glove_file
        glove_dic = {}
        with open(glove_file) as fh:
            lines = fh.readlines()
            for i in tqdm(xrange(len(lines)), desc = "Reading gloVe from disk"):
                arr = line.lstrip().rstrip().split(" ")
                word = arr[0]
                vec =  [float(i) for i in arr[1:]]
                glove_dic[word] = vec
        #get embed_matrix
        embed_matrix = np.empty((len(voc), embed_size))
        #Known
        found_voc = set()
        found_count = 0
        found_vec_avg = np.zeros((embed_size))
        for i in tqdm(xrange(len(voc)), desc = "Vectorize each word in vocabulary"):
            word = voc[i]
            if word in glove_dic:
                embed_matrix[i, :] = glove_dic[word]
                found_voc.add(word)
                found_count += 1
                found_vec_avg += glove_dic[word]
            elif word.lower() in glove_dic:
                embed_matrix[i, :] = glove_dic[word.lower()]
                found_voc.add(word)
                found_count += 1
                found_vec_avg += glove_dic[word.lower()]
            elif word.upper() in glove_dic:
                embed_matrix[i, :] = glove_dic[word.upper()]
                found_voc.add(word)
                found_count += 1
                found_vec_avg += glove_dic[word.upper()]
            else:
                continue
        #UNK
        found_vec_avg /= (1.0 * found_count)
        for i in xrange(len(voc)):
            if voc[i] not in found_voc:
                embed_matrix[i, :] = found_vec_avg
        #self._PAD
        embed_matrix[0, :] = np.zeros((embed_size))

        #save embed_matrix
        np.save(embed_path, embed_matrix)
        print "Embed matrix is save in {}".format(embed_path)
        print "{}/{} tokens in vocabulary has correspondong word vectors".format(found_count, len(voc))
        #73379/102429 tokens in vocabulary has correspondong word vectors

    '''
    Make token_ids using tokens and vocabulary
    '''
    def tokens_to_token_ids(self, voc_file, token_file, token_id_file):
        if os.path.isfile(token_id_file):
            print "{} is ready!".format(token_id_file)
            return
        #load voc
        voc = self.load_vocabulary(voc_file)
        #make rev_voc
        rev_voc = {}
        for i in xrange(len(voc)):
            rev_voc[voc[i]] = i
        with open(token_file) as tf, open(token_id_file, 'w') as idf:
            lines = tf.readlines()
            for i in tqdm(xrange(len(lines)), desc = "Convert tokens in {} to token_ids".format(token_file)):
                token_list = lines[i].decode('utf8').split()
                token_id_list = [str(rev_voc[token.lower()]) for token in token_list]
                idf.write(' '.join(token_id_list) + '\n')
    def tokens_to_token_ids_for_test(self, voc_file, token_file, token_id_file):
        if os.path.isfile(token_id_file):
            print "{} is ready!".format(token_id_file)
            return
        #load voc
        voc = self.load_vocabulary(voc_file)
        #make rev_voc
        rev_voc = {}
        for i in xrange(len(voc)):
            rev_voc[voc[i]] = i
        with open(token_file) as tf, open(token_id_file, 'w') as idf:
            lines = tf.readlines()
            for i in tqdm(xrange(len(lines)), desc = "Convert tokens in {} to token_ids".format(token_file)):
                token_list = lines[i].decode('utf8').split()
                #if vocabulary does not include the token, treat is as self._UNK
                token_id_list = [str(rev_voc[token.lower()]) if token.lower() in rev_voc else str(self.UNK_ID) for token in token_list]
                idf.write(' '.join(token_id_list) + '\n')


    '''
    Get statistics of json file
    '''
    def analyze(self, json_file):
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


        return pass_lengths, ques_lengths


if __name__ == "__main__":
    my_preprocessor = Preprocessor()

    '''Download raw data'''
    my_preprocessor.download("../data/data_raw")



    '''get tokens'''
    my_preprocessor.tokenize_train("../data/data_raw/train-v1.1.json" ,
                                           "../data/data_clean/")
    my_preprocessor.tokenize_test("../data/data_raw/dev-v1.1.json" ,
                                              "../data/data_clean/")

    '''Make voc, rev_voc using train and valid tokens  (!!DO NOT use test tokens)'''
    my_preprocessor.make_voc("../data/data_clean/vocabulary" ,
                            ["../data/data_clean/train.passage", "../data/data_clean/train.question",
                            "../data/data_clean/valid.passage", "../data/data_clean/valid.question"])

    '''Make embedding matrix using vocabulary and glove'''
    my_preprocessor.make_embed("../data/data_clean/vocabulary", "../data/data_raw/glove.6B.100d.txt", 100, "../data/data_clean/vector100")

    # '''Make train ids, valid ids and test ids'''
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/train.passage", "../data/data_clean/train.passage.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/train.question", "../data/data_clean/train.question.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/valid.passage", "../data/data_clean/valid.passage.token_id")
    # my_preprocessor.tokens_to_token_ids("../data/data_clean/vocabulary", "../data/data_clean/valid.question", "../data/data_clean/valid.question.token_id")
    # my_preprocessor.tokens_to_token_ids_for_test("../data/data_clean/vocabulary", "../data/data_clean/test.passage", "../data/data_clean/test.passage.token_id")
    # my_preprocessor.tokens_to_token_ids_for_test("../data/data_clean/vocabulary", "../data/data_clean/test.question", "../data/data_clean/test.question.token_id")

    # '''Get statistics'''
    # my_preprocessor.analyze("../data/data_raw/train-v1.1.json")
    # my_preprocessor.analyze("../data/data_raw/dev-v1.1.json")
