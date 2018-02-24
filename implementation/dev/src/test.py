
import tensorflow as tf
import os
import pickle
import json
import numpy as np

from preprocess import Preprocessor
from util_data import *
from evaluate_v_1_1 import evaluate

def test():
    #process data
    processor = Preprocessor()
    processor.tokenize_test(json_file, dir_to_save)
    passage, passage_mask = pad_token_ids(pass_max_length, token_id_file)
    os.path.join(dir_to_save, 'test.passage')
    os.path.join(dir_to_save, 'test.question')
    os.path.join(dir_to_save, 'test.question_id')


    #revover graph

    #make predictions

    #calculate scores

    #print/save predictions and score












    print "Start testing best graph...."
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    trained_graph = valid_result['graph_path']

    with open(os.path.join(dir_data, test_json_sub_path)) as f:
        data_json = json.load(f)
        dataset = data_json['data']

    #TODO: create sess here and pass sess as parameter to get_json_predictions

    predictions = get_json_predictions(os.path.join(dir_data, test_batches_sub_path), os.path.join(dir_data, test_passage_tokens_sub_path), os.path.join(dir_data, test_question_ids_sub_path), trained_graph)
    score = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}
    test_result = {"graph_path": trained_graph, 'exact_match': score["exact_match"], 'f1': score["f1"], "predictions_path": os.path.join(dir_output, "test_predictions.json")}
    with open(os.path.join(dir_output, "test_predictions.json"), "w") as f:
        f.write(json.dumps(predictions))
    with open(os.path.join(dir_output, "test_result.json"), "w") as f:
        f.write(json.dumps(test_result))
    print "Finish testing best graph...."
    return test_result


'''
batches_file is equal or longer than token_file due to fake datum
'''
#TODO: pass sess as parameter
def get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph):
    '''
        return: json
    '''
    with tf.Session() as sess:
        #TODO: intiialzing sess is necessary? I guess no
        saver = tf.train.import_meta_graph(trained_graph + '.meta')
        saver.restore(sess, trained_graph)
        passage_ph = tf.get_default_graph().get_tensor_by_name("passage_placeholder:0")
        passage_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("passage_sequence_length_placeholder:0")
        ques_ph = tf.get_default_graph().get_tensor_by_name("question_placeholder:0")
        ques_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("question_sequence_length_placeholder:0")
        dist = tf.get_default_graph().get_tensor_by_name("dist:0")
        #get predictions_dist
        pred_dist_list = []
        batches = get_batches(batches_file)
        for batch in batches:
            passage, passage_sequence_length, ques, ques_sequence_length, _ = batch
            pred_dist = sess.run(dist, {passage_ph : passage,
                                        passage_sequence_length_ph : passage_sequence_length,
                                        ques_ph : ques,
                                        ques_sequence_length_ph : ques_sequence_length})
            pred_dist_list.append(pred_dist)
    predictions_dist = np.concatenate(pred_dist_list)#(-1, 2, pass_max_length)

    #turn predictions_dist to prediction
    pred_dic = {}
    with open(passage_tokens_file) as f1, open(question_ids_file) as f2:
        passage_tokens_list = f1.readlines()
        question_id_list = f2.readlines()

    for i in xrange(len(passage_tokens_list)):
        tokens = passage_tokens_list[i].split()
        start = np.argmax(predictions_dist[i][0])
        end = np.argmax(predictions_dist[i][1])
        answer =  ' '.join(tokens[min(start, len(tokens) - 1) : min(end + 1, len(tokens))])
        question_id = question_id_list[i].split()[0]
        pred_dic[question_id] = answer
        # print tokens
        # print answer
        # print
    return pred_dic
