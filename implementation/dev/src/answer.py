import os
import tensorflow as tf
import numpy as np

from preprocess import Preprocessor
from util_data import pad_token_ids, predict_ans_text

def answer(dir_for_ans, pass_strs, ques_strs, pass_max_length, ques_max_length, best_graph_file, voc_file):
    if not os.path.exists(dir_for_ans):
        os.makedirs(dir_for_ans)

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
        print passage_ph
        print passage_mask_ph
        print ques_ph
        print ques_mask_ph
        print beta_s_tensor.shape
        print beta_e_tensor
        print passage
        print passage_mask
        print ques
        print ques_mask
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

if __name__ == "__main__":
    dir_for_ans = "../data/data_ans"
    pass_strs = ["Hunan cuisine, also known as Xiang cuisine, consists of the cuisines of the Xiang River region, Dongting Lake and western Hunan Province in China. It is one of the Eight Great Traditions of Chinese cuisine and is well known for its hot and spicy flavours,[1] fresh aroma and deep colours. Common cooking techniques include stewing, frying, pot-roasting, braising and smoking. Due to the high agricultural output of the region, ingredients for Hunan dishes are many and varied.",\
                "Hunan cuisine, also known as Xiang cuisine, consists of the cuisines of the Xiang River region, Dongting Lake and western Hunan Province in China. It is one of the Eight Great Traditions of Chinese cuisine and is well known for its hot and spicy flavours,[1] fresh aroma and deep colours. Common cooking techniques include stewing, frying, pot-roasting, braising and smoking. Due to the high agricultural output of the region, ingredients for Hunan dishes are many and varied."]
    ques_strs = ["Where does Hunan cuisine come from?", "Where does Hunan cuisine come from?"]
    pass_max_length = 9
    ques_max_length = 5
    best_graph_file = "../output/graphes/March-08-2018-08-32-05"
    voc_file = "../data/data_clean/vocabulary"

    print answer(dir_for_ans, pass_strs, ques_strs, pass_max_length, ques_max_length, best_graph_file, voc_file)
