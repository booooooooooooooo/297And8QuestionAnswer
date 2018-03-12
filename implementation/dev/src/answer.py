import os
import tensorflow as tf
import numpy as np
import json
import sys

from preprocess import Preprocessor
from util_data import pad_token_ids, predict_ans_text, get_data_tuple

def answer(passage, passage_mask, ques, ques_mask, best_graph_file, voc):

    predictions = []
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
        #batch_size = 1 due to limited local memory
        for i in xrange(len(passage)):
            beta_s, beta_e = sess.run([beta_s_tensor, beta_e_tensor], {passage_ph : passage[i: i + 1],
                                                                       passage_mask_ph : passage_mask[i: i + 1],
                                                                       ques_ph : ques[i: i + 1],
                                                                       ques_mask_ph : ques_mask[i: i + 1]})
            #get predicted answers
            idx_s = np.argmax(beta_s, axis=1)
            idx_e = np.argmax(beta_e, axis=1)
            predict_text = predict_ans_text(idx_s, idx_e, passage, voc)
            predictions += predict_text
    return predictions
def get_best_graph(stat_file):
    with open(stat_file) as f:
        stat = json.load(f)

    print "Searching for best graph"
    best_graph = ""

    # best_score = -1.0
    # for batch_stat in stat["train_stat"]:
    #     cur_score = (float(batch_stat["valid_f1"]) + float(batch_stat["valid_em"]) )/ 2.0
    #     if cur_score >= best_score:
    #         best_graph = batch_stat["graph_file"]
    #         best_score = cur_score

    best_valid_loss = sys.float_info.max
    for batch_stat in stat["train_stat"]:
        if float(batch_stat["valid_loss"]) <= best_valid_loss:
            best_graph = batch_stat["graph_file"]
            best_valid_loss = float(batch_stat["valid_loss"])
    print "Have found best graph {}".format(best_graph)
    return best_graph
def prepare_interactive_local(dir_for_ans, pass_strs, ques_strs, pass_max_length, ques_max_length, voc_file):
    #make dir dir_for_ans if it does not exist
    if not os.path.exists(dir_for_ans):
        os.makedirs(dir_for_ans)
    #delete all files in dir_for_ans
    for the_file in os.listdir(dir_for_ans):
        file_path = os.path.join(dir_for_ans, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    #get prep
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

    return passage, passage_mask, ques, ques_mask, voc

def answer_interactive_local(dir_for_ans, stat_file, voc_file):
    with open(stat_file) as f:
        stat = json.load(f)
    pass_max_length = stat["config"]["pass_max_length"]
    ques_max_length = stat["config"]["ques_max_length"]
    best_graph_file = os.path.join(os.path.dirname(stat_file), get_best_graph(stat_file))
    # print os.path.dirname(stat_file)
    # print stat["best_graph"]["graph"]
    # print best_graph_file
    while True:
        passage = raw_input("Enter passage: ")
        ques = raw_input("Enter question: ")
        pass_strs = [passage]
        ques_strs = [ques]
        passage, passage_mask, ques, ques_mask, voc = prepare_interactive_local(dir_for_ans,
                                                 pass_strs, ques_strs, pass_max_length,
                                                 ques_max_length, voc_file)

        print answer(passage, passage_mask, ques, ques_mask, best_graph_file, voc)

def test(dir_data, dir_output, stat_file):
    with open(stat_file) as f:
        stat = json.load(f)
    pass_max_length = stat["config"]["pass_max_length"]
    ques_max_length = stat["config"]["ques_max_length"]
    best_graph_file = os.path.join(os.path.dirname(stat_file), get_best_graph(stat_file))

    #prepare passage, passage_mask, ques, ques_mask
    passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = get_data_tuple("test", dir_data,pass_max_length, ques_max_length)

    #get and save predictions
    predictions = answer(passage, passage_mask, ques, ques_mask, best_graph_file, voc)

    #calculate, print and save test score





    #
    # print "=================="
    # print "Start testing!"
    #
    # #recover best graph
    # print "Recovering best graph"
    # saver = tf.train.import_meta_graph(os.path.join(dir_output, best_graph+ '.meta'))
    # saver.restore(sess, os.path.join(dir_output, best_graph))
    # print "Have Recovered best graph"
    # #get predictions and scores
    # print "Start evaluating!"
    # loss, f1, em, predict_test_answers = self.evaluate(sess, test_data, batch_size, len(test_data[0]))
    #
    # predict_test_answers_file = os.path.join(dir_output, "predict_test_answers-{}".format(datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S")))
    # print "Finish evaluating!"
    # #write to disk and IO
    # print "Writing predict test answers to disk"
    # with open(predict_test_answers_file, 'w') as f:
    #     for ans in predict_test_answers:
    #         f.write(ans + "\n")
    # print "Finish writing!"
    # print "Test loss : {}".format(loss)
    # print "Test f1: {}".format(f1)
    # print "Test em: {}".format(em)
    # print "Test predicted answer saved at {}".format(predict_test_answers_file)
    # stat["test_stat"] = {"loss": str(loss), "f1": str(f1), "em": str(em), "predicted_ans_file": predict_test_answers_file}
    # print "Finish tesing! Congs!"
    # print "=================="

if __name__ == "__main__":

    stat_file = "../output/job57/Stat-March-09-2018-02-06-22"
    # print get_best_graph(stat_file)

    # dir_for_ans = "../data/data_ans"
    # voc_file = "../data/data_clean/vocabulary"
    # answer_interactive_local(dir_for_ans, stat_file, voc_file)

    dir_data = "../data/data_clean"
    dir_output = "../output/local"
    print test(dir_data, dir_output, stat_file)
