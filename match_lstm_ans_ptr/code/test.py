'''
Used for testing.

'''

import argparse
import pickle
import tensorflow as tf
import os

from helper import *
from evaluate_v_1_1 import evaluate

def test(test_json_sub_path, test_passage_tokens_sub_path, test_question_ids_sub_path, test_batches_sub_path, valid_result, dir_data, dir_output):
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
    return test_result
