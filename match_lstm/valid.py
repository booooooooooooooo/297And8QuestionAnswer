'''
Used for validation.

'''
import argparse
import pickle
import tensorflow as tf

from helper import *
from evaluate_v_1_1 import evaluate

def validate(json_file, passage_tokens_file, question_ids_file, batches_file, graph_path_list_file):
    best_exact_match = -1
    best_f1 = -1
    best_graph_path = ""
    with open(graph_path_list_file , 'rb') as f:
        graph_path_list = json.load(f)
    with open(json_file) as f:
        data_json = json.load(f)
        dataset = data_json['data']

    for trained_graph in graph_path_list:
        predictions = get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph)
        score = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}
        #TODO: compare and update in more smart way
        if score["f1"] > best_f1:
            best_f1 = score["f1"]
            best_exact_match = score["exact_match"]
            best_graph_path = trained_graph
    best_graph_info = {"graph_path": best_graph_path, 'exact_match': best_exact_match, 'f1': best_f1}

    return best_graph_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validating tensorflow graph')
    parser.add_argument('json_file')
    parser.add_argument('passage_tokens_file')
    parser.add_argument('question_ids_file')
    parser.add_argument('batches_file')
    parser.add_argument('graph_path_list_file')
    parser.add_argument('best_graph_info_file')
    args = parser.parse_args()


    best_graph_info = validate(args.json_file, args.passage_tokens_file, args.question_ids_file, args.batches_file, args.graph_path_list_file )
    with open(args.best_graph_info_file, 'w') as f:
        f.write(json.dumps(best_graph_info))
