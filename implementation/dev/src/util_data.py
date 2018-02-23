'''
pad/truncate
'''
import numpy as np

from tqdm import tqdm
import random
import os



class DataUtil:
    def __init__(self, PAD_ID):
        self.PAD_ID = PAD_ID

    def pad_token_ids(self, max_len, token_id_file):
        '''
        return
            padded_token_ids: int
            mask: int
        '''
        with open(token_id_file) as f:
            lines = f.readlines()

            padded_token_ids = np.full((len(lines), max_len), self.PAD_ID, dtype = np.int32)
            mask = np.zeros((len(lines),max_len), dtype = np.int32)

            for i in tqdm(xrange(len(lines)), desc = "Pad or truncate token_ids from {}".format(token_id_file)):
                line = lines[i]
                tokens = [int(token) for token in line.split()]
                padded_token_ids[i, 0 : min(max_len, len(tokens))] = tokens[0 : min(max_len, len(tokens))]
                mask[i, 0 : min(max_len, len(tokens))] = np.ones( (min(max_len, len(tokens))), dtype = np.int32)

        return padded_token_ids, mask

    def pad_ans_spans(self, pass_max_len, ans_span_file):
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
