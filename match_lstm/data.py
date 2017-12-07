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
    def getTrain( pass_ques_ans_file, glove_file, batch_s, embed_s):
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
        with open(pass_ques_ans_file) as data_file:
            data = json.load(data_file)
        for article_id in tqdm(xrange(len(data['data'])), desc="Preprocessing {}".format(pass_ques_ans_file)):
            paragraphs = data['data'][article_id]['paragraphs']
            for paragraph_id in xrange(len(paragraphs)):
                context = paragraphs[paragraph_id]['context']
                qas = paragraphs[paragraph_id]['qas']
                print paragraphs[paragraph_id].keys()
                for c in nltk.word_tokenize(context):
                    print c,
                print
                for qas_id in range(len(qas)):
                    question = qas[qas_id]['question']
                    answers = qas[qas_id]['answers']
                    print qas[qas_id].keys()
                    print question
                    for answer_id in xrange(len(answers)):
                        text = answers[answer_id]['text']
                        answer = nltk.word_tokenize(text)
                        print answers[answer_id].keys()
                        print text.encode('utf-8')
                        break
                    break
                break
            break
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
