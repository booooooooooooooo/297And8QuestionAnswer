import numpy as np

from tqdm import tqdm
import random
import os

'''
pad/truncate
'''


class DataUtil:
    def __init__(self):
        self._PAD = b"<pad>"
        self._SOS = b"<sos>"#no _SOS in this project
        self._UNK = b"<unk>"
        self._START_VOCAB = [self._PAD, self._SOS, self._UNK]

        self.PAD_ID = 0
        self.SOS_ID = 1
        self.UNK_ID = 2
    def pad_or_truncate_pass_and_ans(self, pass_token_id_file, pass_max_len, ans_span_file):
        '''
        return
            pass_trim: int
            passage_sequence_length: int
            ans_span_trim: int
        '''

        with open(pass_token_id_file) as pass_f, open(ans_span_file) as ans_f:
            pass_lines = pass_f.readlines()
            ans_lines = ans_f.readlines()

            pass_trim = np.full((len(pass_lines), pass_max_len), self.PAD_ID, dtype = np.int32)
            passage_sequence_length = np.zeros((len(pass_lines),), dtype = np.int32)
            ans_span_trim = np.empty((len(pass_lines), 2), dtype = np.int32)
            for i in tqdm(xrange(len(pass_lines)), desc = "Pad or truncate passage from {} and ans_span from {}".format(pass_token_id_file, ans_span_file)):
                pass_line = pass_lines[i]
                ans_line = ans_lines[i]

                pass_tokens = [int(token) for token in pass_line.split()]
                ans_span = [int(idx) for idx in ans_line.split()]

                pass_trim[i, 0 : min(pass_max_len, len(pass_tokens))] = pass_tokens[0 : min(pass_max_len, len(pass_tokens))]
                passage_sequence_length[i] = min(len(pass_tokens), pass_max_len)
                ans_span_trim[i] = ans_span
                if ans_span_trim[i][1] >= pass_max_len:
                    ans_span_trim[i][1] = pass_max_len - 1
                    if ans_span_trim[i][0] >= pass_max_len:
                        ans_span_trim[i][0] = random.randrange(0, pass_max_len)
            return pass_trim, passage_sequence_length, ans_span_trim



    def pad_or_truncate_itself(self, token_id_file, max_len):
        '''
        return
            trim: int
            sequence_length: int
        '''
        with open(token_id_file) as f:
            lines = f.readlines()

            trim = np.full((len(lines), max_len), self.PAD_ID, dtype = np.int32)
            sequence_length = np.zeros((len(lines),), dtype = np.int32)
            for i in tqdm(xrange(len(lines)), desc = "Pad or truncate token_ids from {}".format(token_id_file)):
                line = lines[i]

                tokens = [int(token) for token in line.split()]

                trim[i, 0 : min(max_len, len(tokens))] = tokens[0 : min(max_len, len(tokens))]
                sequence_length[i] = min(len(tokens), max_len)

        return trim, sequence_length
    def prepare_train(self, pass_token_id_file, pass_max_len, ans_span_file, ques_token_id_file, ques_max_len):
        pass_trim, passage_sequence_length, ans_span_trim = self.pad_or_truncate_pass_and_ans(pass_token_id_file, pass_max_len, ans_span_file)
        ques_trim, ques_sequence_length = self.pad_or_truncate_itself(ques_token_id_file, ques_max_len)
        return pass_trim, passage_sequence_length, ans_span_trim, ques_trim, ques_sequence_length
    def prepare_test(self, pass_token_id_file, pass_max_len, ques_token_id_file, ques_max_len):
        pass_trim, passage_sequence_length = self.pad_or_truncate_itself(pass_token_id_file, pass_max_len)
        ques_trim, ques_sequence_length = self.pad_or_truncate_itself(ques_token_id_file, ques_max_len)
        return pass_trim, passage_sequence_length , ques_trim, ques_sequence_length


if __name__ == "__main__":
    # '''
    # Unit testing
    # '''
    #
    # data_util = DataUtil()
    #
    # pass_token_id_file = "../data/data_clean/train.passage.token_id"
    # pass_max_len = 199
    # ans_span_file = "../data/data_clean/train.answer_span"
    # pass_trim, passage_sequence_length, ans_span_trim = data_util.pad_or_truncate_pass_and_ans(pass_token_id_file, pass_max_len, ans_span_file)
    #
    #
    # ques_token_id_file = "../data/data_clean/train.question.token_id"
    # ques_max_len = 139
    # ques_trim, ques_sequence_length = data_util.pad_or_truncate_itself(ques_token_id_file, ques_max_len)
    #
    #
    # print pass_trim[0]
    # print passage_sequence_length[0]
    # print ans_span_trim[0]
    # print ques_trim[0]
    # print ques_sequence_length[0]
