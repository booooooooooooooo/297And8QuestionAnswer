'''
pad/truncate
'''
import numpy as np

from tqdm import tqdm
import random
import os



def pad_token_ids(max_len, token_id_file, PAD_ID):
    '''
    return
        padded_token_ids: int
        mask: int
    '''
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
        adjusted_ans_span = np.empty((len(lines), 2), dtype = np.int32)
        for i in tqdm(xrange(len(lines)), desc = "Adjust ans_span from {}".format(ans_span_file)):
            line = lines[i]
            ans_span = [int(idx) for idx in line.split()]
            adjusted_ans_span[i] = ans_span
            if adjusted_ans_span[i][1] >= pass_max_len:
                adjusted_ans_span[i][1] = pass_max_len - 1
                if adjusted_ans_span[i][0] >= pass_max_len:
                    adjusted_ans_span[i][0] = pass_max_len - 1
    return adjusted_ans_span

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
def predict_ans_text(idx_s, idx_e, passage, rev_voc):
    predict_text = []
    for i in xrange(len):
        s = idx_s[i]
        e = idx_e[i]
        token_ids = passage[i][s : e + 1]
        tokens = [rev_voc[token_id] for token_id in token_ids]
        predict_text.append(" ".join(tokens))
    return predict_text
def get_data_tuple():
    #TODO
    #shuffle
