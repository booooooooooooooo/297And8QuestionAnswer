'''
Used for testing.

'''

import argparse
import pickle
import tensorflow as tf
import os

from helper import *
from evaluate_v_1_1 import evaluate

def test(test_json_sub_path, test_passage_tokens_sub_path, test_question_ids_sub_path, test_batches_sub_path, valid_result, dir_data,dir_output):
    trained_graph = valid_result['graph_path']

    with open(os.path.join(dir_data, test_json_sub_path)) as f:
        data_json = json.load(f)
        dataset = data_json['data']

    #TODO: create sess here and pass sess as parameter to get_json_predictions

    predictions = get_json_predictions(os.path.join(dir_data, test_batches_sub_path), os.path.join(dir_data, test_passage_tokens_sub_path), os.path.join(dir_data, test_question_ids_sub_path), trained_graph)
    exact_match,  f1 = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}
    test_result = {"graph_path": trained_graph, 'exact_match': exact_match, 'f1': f1, "predictions_path": os.path.join(dir_output, "test_predictions.json")}
    with open(os.path.join(dir_output, "test_predictions.json"), "w") as f:
        f.write(json.dump(predictions))
    return test_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validating tensorflow graph')
    parser.add_argument('json_file')
    parser.add_argument('passage_tokens_file')
    parser.add_argument('question_ids_file')
    parser.add_argument('batches_file')
    parser.add_argument('best_graph_info_file')
    parser.add_argument('test_score_info_file')
    args = parser.parse_args()


    score = test(args.json_file, args.passage_tokens_file, args.question_ids_file, args.batches_file, args.best_graph_info_file)
    with open(args.test_score_info_file, 'w') as f:
        f.write(json.dumps(score))
