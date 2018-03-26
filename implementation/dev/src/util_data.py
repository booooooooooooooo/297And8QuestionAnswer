import numpy as np
from tqdm import tqdm
import random
import os



def pad_token_ids(max_len, token_id_file):
    '''
    return
        padded_token_ids: int
        mask: int
    '''
    PAD_ID = 0

    with open(token_id_file) as f:
        lines = f.readlines()

        padded_token_ids = np.full((len(lines), max_len), PAD_ID, dtype = np.int32)
        mask = np.zeros((len(lines),max_len), dtype = np.int32)

        for i in tqdm(xrange(len(lines)), desc = "Pad or truncate token_ids from {}".format(token_id_file)):
            line = lines[i]
            tokens = [int(token) for token in line.split()]
            padded_token_ids[i, 0 : min(max_len, len(tokens))] = tokens[0 : min(max_len, len(tokens))]
            mask[i, 0 : min(max_len, len(tokens))] = np.ones( (min(max_len, len(tokens))), dtype = np.int32)

    return padded_token_ids, mask

def pad_ans_spans(pass_max_len, ans_span_file):
    '''
    return
        adjusted_ans_span: int
    '''
    with open(ans_span_file) as f:
        lines = f.readlines()
        adjusted_answer_s = np.empty((len(lines),), dtype = np.int32)
        adjusted_answer_e = np.empty((len(lines), ), dtype = np.int32)
        for i in tqdm(xrange(len(lines)), desc = "Adjust ans_span from {}".format(ans_span_file)):
            line = lines[i]
            ans_span = [int(idx) for idx in line.split()]
            adjusted_answer_s[i] = ans_span[0]
            adjusted_answer_e[i] = ans_span[1]
            if adjusted_answer_e[i] >= pass_max_len:
                adjusted_answer_e[i] = pass_max_len - 1
                if adjusted_answer_s[i]>= pass_max_len:
                    adjusted_answer_s[i] = pass_max_len - 1
    return adjusted_answer_s, adjusted_answer_e


def load_text(text_file):
    texts = []
    with open(text_file) as f:
        lines = f.readlines()
        for i in tqdm(xrange(len(lines)), desc = "Loading texts form {}".format(text_file)):
            line = lines[i]
            texts.append(line.rstrip())
    return texts

def load_vocabulary(voc_file):
    voc = []
    with open(voc_file) as f:
        lines = f.readlines()
        for i in tqdm(xrange(len(lines)), desc = "Loading vocabulary form {}".format(voc_file)):
            line = lines[i]
            tokens = line.decode('utf8').split()
            voc.append(tokens[0])
    #make rev_voc
    rev_voc = {}
    for i in xrange(len(voc)):
        rev_voc[voc[i]] = i
    return voc, rev_voc

def get_data_tuple(usage, dir_data, voc_file, pass_max_len, ques_max_length):
    #NOTE: DO NOT shuffle so that predictions can be compared with clean data. Data in dir_data is already shuffled

    passage_token_id_file = os.path.join(dir_data, usage + ".passage.token_id")
    ques_token_id_file = os.path.join(dir_data, usage + ".question.token_id")
    ans_span_file = os.path.join(dir_data, usage + ".answer_span")
    ans_text_file = os.path.join(dir_data, usage + ".answer_text")
    # voc_file = os.path.join(dir_data, "vocabulary")

    passage, passage_mask = pad_token_ids(pass_max_len, passage_token_id_file)
    ques, ques_mask = pad_token_ids(ques_max_length, ques_token_id_file)
    answer_s, answer_e = pad_ans_spans(pass_max_len, ans_span_file)
    answer_text = load_text(ans_text_file)
    voc, _ = load_vocabulary(voc_file)

    return passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc


def get_batches(passage, passage_mask, ques, ques_mask, answer_s, answer_e, batch_size):
    batches = []

    #shuffle
    indices = range(len(passage))
    np.random.shuffle(indices)

    for num in xrange( len(passage) / batch_size ):
        batches.append([ [passage[i] for i in indices[num * batch_size : (num + 1) * batch_size]],
                         [passage_mask[i] for i in indices[num * batch_size : (num + 1) * batch_size]],
                         [ques[i] for i in indices[num * batch_size : (num + 1) * batch_size]],
                         [ques_mask[i] for i in indices[num * batch_size : (num + 1) * batch_size]],
                         [answer_s[i] for i in indices[num * batch_size : (num + 1) * batch_size]],
                         [answer_e[i] for i in indices[num * batch_size : (num + 1) * batch_size]] ])

    return batches
def predict_ans_text(idx_s, idx_e, passage, voc):
    predict_text = []
    for i in xrange(len(idx_s)):
        s = idx_s[i]
        e = idx_e[i]
        token_ids = passage[i][s : e + 1]
        tokens = [voc[token_id] for token_id in token_ids]
        predict_text.append(" ".join(tokens))
    return predict_text
