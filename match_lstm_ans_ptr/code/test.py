'''
Used for testing.

'''

import argparse
import pickle
import tensorflow as tf

from helper import *
from evaluate_v_1_1 import evaluate

def test(json_file, passage_tokens_file, question_ids_file, batches_file, best_graph_info_file):
    with open(best_graph_info_file , 'rb') as f:
        best_graph_info = json.load(f)
    trained_graph = best_graph_info['graph_path']

    with open(json_file) as f:
        data_json = json.load(f)
        dataset = data_json['data']

    #TODO: create sess here and pass sess as parameter to get_json_predictions
    
    predictions = get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph)
    score = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}

    return score

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
