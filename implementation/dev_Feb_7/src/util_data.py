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

    def getTrain(self, dir_data, pass_max_len, ques_max_len):
        pass_token_id_file= os.path.join(dir_data, "train.passage.token_id")
        ans_span_file=os.path.join(dir_data, "train.answer_span")
        ques_token_id_file=os.path.join(dir_data, "train.question.token_id")

        passage, passage_sequence_length, ans_span_trim = self.pad_or_truncate_pass_and_ans(pass_token_id_file, pass_max_len, ans_span_file)
        ques, ques_sequence_length = self.pad_or_truncate_itself(ques_token_id_file, ques_max_len)
        return passage, passage_sequence_length, ques, ques_sequence_length, ans_span_trim

    def getTrainBatches(self, dir_data, pass_max_len, ques_max_len, batch_size, keep_prob):
        print "Start reading train batches from {}".format(dir_data)
        pass_token_id_file = os.path.join(dir_data, "train.passage.token_id")
        ans_span_file = os.path.join(dir_data, "train.answer_span")
        ques_token_id_file = os.path.join(dir_data, "train.question.token_id")
        passage, passage_sequence_length, ques, ques_sequence_length, ans_span = self.getTrain(dir_data, pass_max_len, ques_max_len)
        batches = []
        for i in xrange(len(passage) / batch_size):
            batch_passage = passage[i * batch_size : (i + 1) * batch_size]
            batch_passage_sequence_length = passage_sequence_length[i * batch_size : (i + 1) * batch_size]
            batch_ques = ques[i * batch_size : (i + 1) * batch_size]
            batch_ques_sequence_length = ques_sequence_length[i * batch_size : (i + 1) * batch_size]
            batch_ans_span = ans_span[i * batch_size : (i + 1) * batch_size]
            batches.append([batch_passage, batch_passage_sequence_length, batch_ques, batch_ques_sequence_length, batch_ans_span, keep_prob])
        #ignore the remain data
        print "Finish reading train batches"
        return batches
    def getTexts(self, text_file):
        with open(text_file) as f:
            lines = f.readlines()
            result = [line.strip("\n") for line in lines]
        return result
    def getVoc(self, voc_file):
        with open(voc_file) as f:
            lines = f.readlines()
            voc = [line.strip('\n') for line in lines]
        rev_voc = {}
        for i in xrange(len(voc)):
            rev_voc[voc[i]] = i
        return voc, rev_voc
    def getValid(self, dir_data, pass_max_len, ques_max_len):
        print "Start reading validation data from {}".format(dir_data)
        pass_token_id_file=os.path.join(dir_data, "valid.passage.token_id")
        ans_span_file=os.path.join(dir_data, "valid.answer_span")
        ques_token_id_file=os.path.join(dir_data, "valid.question.token_id")
        pass_text_file=os.path.join(dir_data, "valid.passage")
        ques_text_file=os.path.join(dir_data, "valid.question")
        ans_text_file=os.path.join(dir_data, "valid.answer_text")
        voc_file=os.path.join(dir_data, "vocabulary")
        passage, passage_sequence_length, ans = self.pad_or_truncate_pass_and_ans(pass_token_id_file, pass_max_len, ans_span_file)
        ques, ques_sequence_length = self.pad_or_truncate_itself(ques_token_id_file, ques_max_len)
        passage_text=self.getTexts(pass_text_file)
        ques_text=self.getTexts(ques_text_file)
        ans_texts=self.getTexts(ans_text_file)
        voc, rev_voc = self.getVoc(voc_file)
        print "Finish reading validation data"


        return passage, passage_sequence_length, ques, ques_sequence_length, ans, passage_text, ques_text, ans_texts, voc

if __name__ == "__main__":
    # '''
    # Unit testing
    # '''
    #
    datautil = DataUtil()

    dir_data = "../data/data_clean"
    pass_max_len = 400
    ques_max_len = 60
    batch_size = 77
    keep_prob = 0.9

    passage, passage_sequence_length, ques, ques_sequence_length, ans, passage_text, ques_text, ans_texts, voc = datautil.getValid(dir_data, pass_max_len, ques_max_len)
    print passage_text[0]
    print ques_text[0]
    print ans_texts[0]
    print voc[4]

    batches = datautil.getTrainBatches(dir_data, pass_max_len, ques_max_len, batch_size, keep_prob)
    print len(batches)
    # passage, passage_sequence_length, ques, ques_sequence_length,ans_span, _ = batches[1]
    # print passage
    # print passage_sequence_length
    # print ques
    # print ques_sequence_length
    # print ans_span
