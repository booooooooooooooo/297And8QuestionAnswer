import tensorflow as tf
import pickle
import json
import numpy as np

def get_batches(batches_file, small_size = False):

    print "Reading batched data from disk......."
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)
    if not small_size:
        return batches
    else:
        return batches[0 : len(batches) / 200]

def get_json_predictions(batches_file, token_file, trained_graph):
    '''
        return: json
    '''
    #restore graph
    sess = tf.Session()
    #TODO: intiialzing sess is necessary? I guess no
    saver = tf.train.import_meta_graph(trained_graph + '.meta')
    saver.restore(sess, target_graph)
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
                                                          ques_sequence_length_ph : ques_sequence_length)
        pred_dist_list.append(pred_dist)
    predictions_dist = np.stack(pred_dist_list)#(#_of_data, 2, pass_max_length)

    #turn predictions_dist to prediction
    pred_dic = {}
    with open(token_file) as f:
        token_list = f.readlines()
    print len(token_list)
    print predictions_dist
    for i in xrange(token_list):
        tokens = token_list[i].split()
        answer_id = tokens[0]
        start = predictions_dist[i][0].index(max(predictions_dist[i][0]))
        end = predictions_dist[i][1].index(max(predictions_dist[i][1]))
        answer = token_list[i][start + 1, end + 1 + 1]
        pred_dic[answer_id] = answer
    predictions = json.dumps(pred_dic)

    return predictions
