'''
Used for validation.

'''
import argparse
import pickle
import tensorflow as tf
import os

from helper import *
from evaluate_v_1_1 import evaluate

def valid(valid_json_sub_path, valid_passage_tokens_sub_path, valid_question_ids_sub_path, valid_batches_sub_path, graph_sub_paths_list, dir_data, dir_output ):
    print "Start validating trained graphes...."
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    #TODO: create sess here and pass sess as parameter to get_json_predictions
    best_exact_match = -1
    best_f1 = -1
    best_graph_path = ""
    best_predictions = {}

    with open(os.path.join(dir_data, valid_json_sub_path)) as f:
        data_json = json.load(f)
        dataset = data_json['data']
    for trained_graph in graph_sub_paths_list:
        predictions = get_json_predictions(os.path.join(dir_data, valid_batches_sub_path), os.path.join(dir_data, valid_passage_tokens_sub_path), os.path.join(dir_data, valid_question_ids_sub_path), trained_graph)
        score = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}
        #TODO: compare and update in more smart way
        if score["f1"] > best_f1:
            best_f1 = score["f1"]
            best_exact_match = score["exact_match"]
            best_graph_path = trained_graph
            best_predictions = predictions
    valid_result = {"graph_path": best_graph_path, 'exact_match': best_exact_match, 'f1': best_f1, "predictions_path": os.path.join(dir_output, "valid_predictions.json")}

    with open(os.path.join(dir_output, "valid_predictions.json"), 'w') as f:
        f.write(json.dumps(best_predictions))
    with open(os.path.join(dir_output, "valid_result.json"), 'w') as f:
        f.write(json.dumps(valid_result))
    print "Finish validating trained graphes...."
    return valid_result
