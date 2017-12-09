'''
Used to get word embedding, get vocabulary, get mini_batches etc.
'''

import os
import json
from tqdm import tqdm
import nltk

class Data:
    def __init__(self):
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.PAD = unicode("_pad")
        self.UNK = unicode("_unk")
    def tokenize(self, s):
        '''
        paras
            s: unicode
        return
            s_token: list of unicode
        '''
        s_token = nltk.word_tokenize(s)
        return [token.decode('utf-8') for token in s_token]

    def c_id_to_token_id(self, context, context_token):
        '''
        paras
            context: unicode
            context_token: list of unicode
        '''
        c_id_to_token_id_map = {}
        token_id = 0
        id_in_token = 0
        for c_id, c in enumerate(context):
            if nltk.word_tokenize(c) != []:
                if id_in_token == len(context_token[token_id]):
                    token_id += 1
                    id_in_token = 0
                c_id_to_token_id_map[c_id] = token_id
                id_in_token += 1

        return c_id_to_token_id_map


    def getTrain(self, pass_ques_ans_file, glove_file, batch_s, embed_s):
        '''
        return
            vocabulary (index: token)
            embed_matrix (index:vector), 0: 0 vector
            list of (batch_pass, batch_pass_rev, batch_pass_mask, batch_ques, batch_ques_mask, batch_ans)
                batch_pass: index in vocabulary (None, pass_max_l) padding with 0
                batch_ques: index in vocabulary (None, ques_max_l) padding with 0
                batch_ans: one-hot or zero-hot in pass (None, 2, pass_max_l)
            pass_max_l
            ques_max_l
        '''
        #convert json to list of tokens
        '''
        input
            pass_ques_ans_file
        output
            data = [([context_tokens], [ques_tokens], [ans_id_in_context_tokens])]
            pass_max_l
            ques_max_l
        '''
        data_token = []
        with open(pass_ques_ans_file) as fh:
            data_json = json.load(fh)
        for article_id in tqdm(xrange(len(data_json['data'])), desc="Preprocessing {}".format(pass_ques_ans_file)):
            paragraphs = data['data'][article_id]['paragraphs']
            for paragraph_id in xrange(len(paragraphs)):
                context = paragraphs[paragraph_id]['context']
                context = context.decode('utf-8')
                context_token = self.tokenize(context)
                # print context.encode('utf-8')
                # print [token.decode('utf-8') for token in context_token]
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
                        datum = (context_token, question_token, [a_s, a_e])
                        data_token.append(datum)
                        # print text.encode('utf-8')
                        # print [word.encode('utf-8') for word in context_token[a_s : a_e + 1]]

        # vocabulary
        '''
        input
            data = [([context_tokens], [ques_tokens], [ans_id_in_context_tokens])]
        output
            vocabulary = {0:"_pad", 1: "_unk", voc_id:token}
        '''
        # embed_matrix
        '''
        input
            glove_file
            vocabulary = {0:"_pad", 1: "_unk", voc_id:token}
        output
            embed_matrix = |row 0: 0->, row 1:0->, row 2: ?->,...|
            Note: "_pad" and "_unk" correspond to zero vectors
            known_vocabulary
        '''
        # token to id
        '''
        output
            data_id = [([context_id_in_voc], [ques_id_in_voc], [ans_id_in_context_tokens])]
            Note: token not having glove is regarded as "_unk"
        '''
        # padding
        '''
        output
            data_id_padded
        '''
        # batching
        '''
        output
            feed ready batch_data
        '''


    def getPredict( pass_ques_file, glove_file, embed_s, batch_s):
        #TODO return: pass_l, ques_l, vocabulary, vocabulary_rev, embed_matrix, batch_data
        return
