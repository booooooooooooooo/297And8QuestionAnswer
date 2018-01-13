import nltk
import json
from tqdm import tqdm
import os
import sys
import argparse

'''
nltk.word_tokenize has some weird behaviours.
First, it tokenizes ''word'' to "word". This case is not corrected in this code.
Second, it tokenizes "word" to ``word''. This caseis corrected in this code.
'''

class Preprocessor:
    def analyze(self, json_file):
        count = 0
        pass_max_length = 0
        pass_ave_length = 0
        ques_max_length = 0
        ques_ave_length = 0

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
                        count += 1
                        pass_max_length = max(pass_max_length, len(context_token))
                        pass_ave_length += len(context_token)
                        ques_max_length = max(ques_max_length, len(question_token))
                        ques_ave_length += len(question_token)
        pass_ave_length /= count
        ques_ave_length /= count
        print "How many (passage, question, answer) tuples : {} \n pass_max_length: {} \n pass_ave_length: {}\n ques_max_length: {} \n ques_ave_length :{} \n".format(count, pass_max_length, pass_ave_length, ques_max_length, ques_ave_length)

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

    def get_token_with_answers(self, json_file, dir_to_save, prefix):
        if os.path.isfile(os.path.join(dir_to_save, prefix, ".passage" )):
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
        if not os.path.isfile(os.path.join(dir_to_save, prefix, '.passage')):#all exist or none exists
            with open(os.path.join(dir_to_save, prefix, '.passage'), 'w') as passage_file, \
                 open(os.path.join(dir_to_save, prefix, '.question'), 'w') as question_file, \
                 open(os.path.join(dir_to_save, prefix, '.answer_text'), 'w') as ans_text_file, \
                 open(os.path.join(dir_to_save, prefix, '.answer_span'), 'w') as ans_span_file:
                 for i in tqdm(xrange(len(passage_list)), desc="Writing {} tokens to {}".format(prefix, dir_to_save)):
                     passage_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                     question_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                     ans_text_file.write(answer_text_list[i].encode('utf8') + '\n')
                     ans_span_file.write(' '.join(answer_span_list[i]) + '\n')
        else:
            print "{} tokens has already been retrieved from \"{}\".".format(prefix, json_file)

    def get_token_without_answers(self, json_file, dir_to_save, prefix):
        if os.path.isfile(os.path.join(dir_to_save, prefix, ".passage" )):
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
        if not os.path.isfile(os.path.join(dir_to_save, prefix, '.passage')):#all exist or none exists
            with open(os.path.join(dir_to_save, prefix, '.passage'), 'w') as passage_file, \
                 open(os.path.join(dir_to_save, prefix, '.question'), 'w') as question_file, \
                 open(os.path.join(dir_to_save, prefix, '.question_id'), 'w') as question_id_file:
                 for i in tqdm(xrange(len(passage_list)), desc="Writing my {} tokens to {} folder".format(prefix, dir_to_save)):
                     passage_file.write(' '.join([token.encode('utf8') for token in passage_list[i]]) + '\n')
                     question_file.write(' '.join([token.encode('utf8') for token in question_list[i]]) + '\n')
                     question_id_file.write(question_id_list[i].encode('utf8') + '\n')
        else:
            print "{} tokens has already been retrieved from \"{}\".".format(prefix, json_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing json to tokens')
    parser.add_argument('function_name')
    parser.add_argument('parameters', metavar='para', type=str, nargs='+',
                        help='sequence of parameters')
    args = parser.parse_args()

    my_preprocessor = Preprocessor()

    getattr(my_preprocessor, args.function_name)(*args.parameters)
