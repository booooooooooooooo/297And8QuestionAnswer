import nltk
import json
from tqdm import tqdm
import os
import sys



class Preprocessor:
    def tokenize(self, s):
        '''
        paras
            s: unicode
        return
            s_token: list of unicode
        '''
        s_token = nltk.word_tokenize(s)
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
                if id_in_cur_token == len(context_token[token_id]):
                    token_id += 1
                    id_in_cur_token = 0
                c_id_to_token_id_map[c_id] = token_id
                id_in_cur_token += 1

        return c_id_to_token_id_map



    def preprocess_train(self, pass_ques_ans_json_path, dir_to_save):
        '''
        Splits out pass, ques and ans from json file.

        paras
            pass_ques_ans_json_path
            dir_to_save

        '''
        with open(pass_ques_ans_json_path) as fh:
            data_json = json.load(fh)

        with open(os.path.join(dir_to_save, 'passage'), 'w') as passage_file, \
             open(os.path.join(dir_to_save, 'question'), 'w') as question_file, \
             open(os.path.join(dir_to_save, 'ans_text'), 'w') as ans_text_file, \
             open(os.path.join(dir_to_save, 'ans_apan'), 'w') as ans_span_file:
            for article_id in tqdm(xrange(len(data_json['data'])), desc="Preprocessing {}".format(pass_ques_ans_json_path)):
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
                            #write to file
                            passage_file.write(' '.join([token.encode('utf8') for token in context_token]) + '\n')
                            question_file.write(' '.join([token.encode('utf8') for token in question_token]) + '\n')
                            ans_text_file.write(text.encode('utf8') + '\n')
                            ans_span_file.write(' '.join([str(a_s), str(a_e)]) + '\n')
                            # print context_token
                            # print question_token
                            # print text
                            # print [a_s, a_e]

    def analyze(self, pass_ques_ans_json_path):
        count = 0
        pass_max_length = 0
        pass_ave_length = 0
        ques_max_length = 0
        ques_ave_length = 0

        with open(pass_ques_ans_json_path) as fh:
            data_json = json.load(fh)
        for article_id in tqdm(xrange(len(data_json['data'])), desc="Analyzing {}".format(pass_ques_ans_json_path)):
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





#
# reload(sys)
# sys.setdefaultencoding('utf8')
#
# def tokenize(sequence):
#     tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
#     return map(lambda x:x.encode('utf8'), tokens)
#
#
# def c_id_to_token_id(context, context_tokens):
#     acc = ''
#     current_token_idx = 0
#     token_map = dict()
#
#     for char_idx, char in enumerate(context):
#         if char != u' ':
#             acc += char
#             context_token = unicode(context_tokens[current_token_idx])
#             if acc == context_token:
#                 syn_start = char_idx - len(acc) + 1
#                 token_map[syn_start] = [acc, current_token_idx]
#                 acc = ''
#                 current_token_idx += 1
#     return token_map
#
# def preprocess_train(pass_ques_ans_json_path):
#     with open(pass_ques_ans_json_path) as fh:
#         dataset = json.load(fh)
#     prefix = "./data_token/"
#     tier = "train"
#
#     """Reads the dataset, extracts context, question, answer,
#     and answer pointer in their own file. Returns the number
#     of questions and answers processed for the dataset"""
#     qn, an = 0, 0
#     skipped = 0
#
#     with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
#          open(os.path.join(prefix, tier +'.question'), 'w') as question_file,\
#          open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
#          open(os.path.join(prefix, tier +'.span'), 'w') as span_file:
#
#         for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
#             article_paragraphs = dataset['data'][articles_id]['paragraphs']
#             for pid in range(len(article_paragraphs)):
#                 context = article_paragraphs[pid]['context']
#                 # The following replacements are suggested in the paper
#                 # BidAF (Seo et al., 2016)
#                 context = context.replace("''", '" ')
#                 context = context.replace("``", '" ')
#
#                 context_tokens = tokenize(context)
#                 answer_map = c_id_to_token_id(context, context_tokens)
#
#                 qas = article_paragraphs[pid]['qas']
#                 for qid in range(len(qas)):
#                     question = qas[qid]['question']
#                     question_tokens = tokenize(question)
#
#                     answers = qas[qid]['answers']
#                     qn += 1
#
#                     num_answers = range(1)
#
#                     for ans_id in num_answers:
#                         # it contains answer_start, text
#                         text = qas[qid]['answers'][ans_id]['text']
#                         a_s = qas[qid]['answers'][ans_id]['answer_start']
#
#                         text_tokens = tokenize(text)
#
#                         answer_start = qas[qid]['answers'][ans_id]['answer_start']
#
#                         answer_end = answer_start + len(text)
#
#                         last_word_answer = len(text_tokens[-1]) # add one to get the first char
#
#                         try:
#                             a_start_idx = answer_map[answer_start][1]
#
#                             a_end_idx = answer_map[answer_end - last_word_answer][1]
#
#                             # remove length restraint since we deal with it later
#                             context_file.write(' '.join(context_tokens) + '\n')
#                             question_file.write(' '.join(question_tokens) + '\n')
#                             text_file.write(' '.join(text_tokens) + '\n')
#                             span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
#
#                         except Exception as e:
#                             skipped += 1
#
#                         an += 1
#
#     print("Skipped {} question/answer pairs in {}".format(skipped, tier))
#     return qn,an
