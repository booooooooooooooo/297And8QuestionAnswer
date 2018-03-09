import os
import glob
import tensorflow as tf
import numpy as np
import json

from preprocess import Preprocessor
from util_data import pad_token_ids, predict_ans_text

def answer(dir_for_ans, pass_strs, ques_strs, pass_max_length, ques_max_length, best_graph_file, voc_file):
    if not os.path.exists(dir_for_ans):
        os.makedirs(dir_for_ans)
    files = glob.glob(dir_for_ans)
    for f in files:
        os.remove(f)

    prep = Preprocessor()
    #load voc and rev_voc
    voc, rev_voc = prep.load_vocabulary(voc_file)
    #get pass_tokens and ques_tokens
    pass_token_file = os.path.join(dir_for_ans, "passage.token")
    ques_token_file = os.path.join(dir_for_ans, "ques.token")
    with open(pass_token_file, "w") as pf, open(ques_token_file, "w") as qf:
        for i in xrange(len(pass_strs)):
            pf.write(' '.join(prep.tokenize(pass_strs[i])) + "\n")
            qf.write(' '.join(prep.tokenize(ques_strs[i])) + "\n")
    #get pass_ids and ques_ids
    pass_id_file = os.path.join(dir_for_ans, "passage.id")
    ques_id_file = os.path.join(dir_for_ans, "ques.id")
    prep.tokens_to_token_ids(voc_file, pass_token_file, pass_id_file)
    prep.tokens_to_token_ids(voc_file, ques_token_file, ques_id_file)
    #get passage_mask and question_mask
    passage, passage_mask = pad_token_ids(pass_max_length, pass_id_file)
    ques, ques_mask = pad_token_ids(ques_max_length, ques_id_file)
    #create sess
    with tf.Session() as sess:
        #restore graph
        saver = tf.train.import_meta_graph(best_graph_file + '.meta')
        saver.restore(sess, best_graph_file)
        passage_ph = tf.get_default_graph().get_tensor_by_name("passage_placeholder:0")
        passage_mask_ph = tf.get_default_graph().get_tensor_by_name("passage_mask_placeholder:0")
        ques_ph = tf.get_default_graph().get_tensor_by_name("question_placeholder:0")
        ques_mask_ph = tf.get_default_graph().get_tensor_by_name("question_mask_placeholder:0")
        beta_s_tensor = tf.get_default_graph().get_tensor_by_name("beta_s:0")
        beta_e_tensor = tf.get_default_graph().get_tensor_by_name("beta_e:0")
        # print passage_ph
        # print passage_mask_ph
        # print ques_ph
        # print ques_mask_ph
        # print beta_s_tensor.shape
        # print beta_e_tensor
        # print passage
        # print passage_mask
        # print ques
        # print ques_mask
        #get beta_s and beta_e
        #TODO: batching feed
        beta_s, beta_e = sess.run([beta_s_tensor, beta_e_tensor], {passage_ph : passage,
                                                                   passage_mask_ph : passage_mask,
                                                                   ques_ph : ques,
                                                                   ques_mask_ph : ques_mask})
        #get predicted answers
        idx_s = np.argmax(beta_s, axis=1)
        idx_e = np.argmax(beta_e, axis=1)
        predict_text = predict_ans_text(idx_s, idx_e, passage, voc)
    return predict_text
def answer_static_floyd():
    #TODO
    return
def answer_interactive_local(dir_for_ans, stat_file, voc_file):
    with open(stat_file) as f:
        stat = json.load(f)
    pass_max_length = stat["config"]["pass_max_length"]
    ques_max_length = stat["config"]["ques_max_length"]
    best_graph_file = os.path.join(os.path.dirname(stat_file), stat["best_graph"]["graph"] )
    # print os.path.dirname(stat_file)
    # print stat["best_graph"]["graph"]
    # print best_graph_file
    while True:
        passage = raw_input("Enter passage: ")
        ques = raw_input("Enter question: ")
        pass_strs = [passage]
        ques_strs = [ques]
        print answer(dir_for_ans, pass_strs, ques_strs, pass_max_length, ques_max_length, best_graph_file, voc_file)
if __name__ == "__main__":
    dir_for_ans = "../data/data_ans"
    stat_file = "../output/cWxPUnSpPLS2EmjkvdQHib/Stat-March-09-2018-01-21-20"
    voc_file = "../data/data_clean/vocabulary"
    answer_interactive_local(dir_for_ans, stat_file, voc_file)
