import os
import tensorflow as tf
import numpy as np
import json
import sys
from tqdm import tqdm
import datetime
import argparse

from preprocess import Preprocessor
from util_data import pad_token_ids, predict_ans_text, get_data_tuple
from evaluate_v_1_1 import f1_score, exact_match_score, evaluate


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

        #NOTE: for quick unit test; TODO: change to xrange(len(passage))
        # for i in tqdm( xrange(min(len(passage), 10)), desc = "Calcualte predictions"):
        for i in tqdm( xrange(len(passage)), desc = "Calcualte predictions"):#batch_size = 1 due to limited local memory
            beta_s, beta_e = sess.run([beta_s_tensor, beta_e_tensor], {passage_ph : passage[i: i + 1],
                                                                       passage_mask_ph : passage_mask[i: i + 1],
                                                                       ques_ph : ques[i: i + 1],
                                                                       ques_mask_ph : ques_mask[i: i + 1]})
            #get predicted answers
            idx_s = np.argmax(beta_s, axis=1)
            idx_e = np.argmax(beta_e, axis=1)
            predict_text = predict_ans_text(idx_s, idx_e, passage[i:i+1], voc)
            predictions += predict_text
            # for p in passage[i:i+1]:
            #     print [voc[token_id] for token_id in p]
            # for p in ques[i:i+1]:
            #     print [voc[token_id] for token_id in p]
            # print predict_text
    return predictions
def get_best_graph(stat_file):
    with open(stat_file) as f:
        stat = json.load(f)

    print "Searching for best graph"
    best_graph = ""

    # #use score to select best graph
    # best_score = -1.0
    # for batch_stat in stat["train_stat"]:
    #     cur_score = (float(batch_stat["valid_f1"]) + float(batch_stat["valid_em"]) )/ 2.0
    #     if cur_score >= best_score:
    #         best_graph = batch_stat["graph_file"]
    #         best_score = cur_score

    # #use valid loss to select best graph
    # best_valid_loss = sys.float_info.max
    # for batch_stat in stat["train_stat"]:
    #     if float(batch_stat["valid_loss"]) <= best_valid_loss:
    #         best_graph = batch_stat["graph_file"]
    #         best_valid_loss = float(batch_stat["valid_loss"])
    # print "Have found best graph {}".format(best_graph)

    #use last graph as best graph
    epoch = -1
    batch = -1
    for batch_stat in stat["train_stat"]:
        if int(batch_stat["epoch"]) > epoch or (int(batch_stat["epoch"]) >= epoch and int(batch_stat["batch"]) > batch):
            best_graph = batch_stat["graph_file"]
            epoch = int(batch_stat["epoch"])
            batch = int(batch_stat["batch"])

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

    passage = raw_input("Enter passage: ")
    ques = raw_input("Enter question: ")
    pass_strs = [passage]
    ques_strs = [ques]
    passage, passage_mask, ques, ques_mask, voc = prepare_interactive_local(dir_for_ans,
                                             pass_strs, ques_strs, pass_max_length,
                                             ques_max_length, voc_file)

    print answer(passage, passage_mask, ques, ques_mask, best_graph_file, voc)

def get_test_predictions(dir_test_data, voc_file, dir_output, stat_file):
    with open(stat_file) as f:
        stat = json.load(f)
    pass_max_length = stat["config"]["pass_max_length"]
    ques_max_length = stat["config"]["ques_max_length"]
    best_graph_file = os.path.join(os.path.dirname(stat_file), get_best_graph(stat_file))

    #prepare passage, passage_mask, ques, ques_mask
    passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = get_data_tuple("test", dir_test_data, voc_file, pass_max_length, ques_max_length)
    # passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text = [data[0:20] for data in (passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text)]

    #get and save predictions
    predictions = answer(passage, passage_mask, ques, ques_mask, best_graph_file, voc)
    # with open(os.path.join(dir_output, "test_predictions" + datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S")), 'w') as f:
    #     for pred in predictions:
    #         f.write(pred + "\n")
    return answer_text, predictions

def test_on_official(dir_test_data, voc_file, dir_output, stat_file, arch):
    answer_text, predictions = get_test_predictions(dir_test_data, voc_file, dir_output, stat_file)
    with open(os.path.join(dir_output, "test_predictions_" + arch + "_" + datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S")), 'w') as f:
        for predict_text in predictions:
            f.write(predict_text + "\n");

    #Test score on unique answer
    f1 = 0.0
    em = 0.0
    for j in xrange(len(predictions)):
        f1 += f1_score(predictions[j].decode('utf8'), answer_text[j].decode('utf8'))
        em += exact_match_score(predictions[j].decode('utf8'), answer_text[j].decode('utf8'))
    f1 /= len(predictions)
    em /= len(predictions)
    test_score = {"f1": str(f1), "em": str(em)}
    print "Test score on unique answer: "
    print test_score

    #Test score on multiple answers
    question_id_file = os.path.join(dir_test_data, "test" + ".question_id")
    with open(question_id_file) as f:
        question_id_list = [line.rstrip() for line in f.readlines()]
    predictions = dict(zip(question_id_list, predictions))
    with open(os.path.join(dir_test_data, "dev-v1.1.json")) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    test_score_multiple = evaluate(dataset, predictions)
    print "Test score on multiple answers: "
    print test_score_multiple


    with open(os.path.join(dir_output, "test_score_" + arch + "_" + datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S")), 'w') as f:
        f.write(json.dumps({"unique_answer": test_score, "multiple_answers": test_score_multiple }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("machine", choices=['local', 'floyd'])
    args = parser.parse_args()

    if args.machine == 'local':
        dir_for_ans = "../data/data_ans"
        voc_file = "../data/data_clean/vocabulary"

        stat_file_match = "../output/job78/Stat-March-31-2018-02-16-09"
        # stat_file_match_change1 = "../output/job71/Stat-March-22-2018-11-55-08"
        # stat_file_match_change2 = "../output/job81/Stat-April-05-2018-05-02-49"
        # stat_file_match_change3 = "../output/job75/Stat-March-29-2018-01-45-17"

        answer_interactive_local(dir_for_ans, stat_file_match, voc_file)
        # answer_interactive_local(dir_for_ans, stat_file_match_change1, voc_file)
        # answer_interactive_local(dir_for_ans, stat_file_match_change2, voc_file)
        # answer_interactive_local(dir_for_ans, stat_file_match_change3, voc_file)

    if args.machine == "floyd":
        '''
dir_test_data="bo.nov29/datasets/squad_test/1"
dir_voc="bo.nov29/datasets/squad/5"
dir_match="bo.nov29/datasets/output_job78/1"
dir_match_change1="bo.nov29/datasets/output_job71/1"
dir_match_change2="bo.nov29/datasets/output_job81/1"
dir_match_change3="bo.nov29/datasets/output_job75/1"
        '''
        dir_test_data = "/dir_test_data"
        voc_file = "/dir_voc/vocabulary"
        dir_output = "/output"
        stat_file_match = "/dir_match/Stat-March-31-2018-02-16-09"
        stat_file_match_change1 = "/dir_match_change1/Stat-March-22-2018-11-55-08"
        stat_file_match_change2 = "/dir_match_change2/Stat-April-05-2018-05-02-49"
        stat_file_match_change3 = "/dir_match_change3/Stat-March-29-2018-01-45-17"

        '''
        floyd run --env tensorflow-1.4:py2  --data $dir_test_data:/dir_test_data --data $dir_voc:/dir_voc --data $dir_match:/dir_match "python answer.py floyd"
        '''

        # print "Testing on match"
        # test_on_official(dir_test_data, voc_file, dir_output, stat_file_match, "match")

        '''
        floyd run --env tensorflow-1.4:py2  --data $dir_test_data:/dir_test_data --data $dir_voc:/dir_voc --data $dir_match_change1:/dir_match_change1 "python answer.py floyd"
        '''

        # print "Testing on match_change1"
        # test_on_official(dir_test_data, voc_file, dir_output, stat_file_match_change1, "match_change1")

        '''
        floyd run --env tensorflow-1.4:py2  --data $dir_test_data:/dir_test_data --data $dir_voc:/dir_voc --data $dir_match_change2:/dir_match_change2 "python answer.py floyd"
        '''

        # print "Testing on match_change2"
        # test_on_official(dir_test_data, voc_file, dir_output, stat_file_match_change2, "match_change2")

        '''
        floyd run --env tensorflow-1.4:py2  --data $dir_test_data:/dir_test_data  --data $dir_voc:/dir_voc --data $dir_match_change3:/dir_match_change3 "python answer.py floyd"
        '''

        print "Testing on match_change3"
        test_on_official(dir_test_data, voc_file, dir_output, stat_file_match_change3, "match_change3")
