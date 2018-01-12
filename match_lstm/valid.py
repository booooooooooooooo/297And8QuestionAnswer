'''
Used for validation.

TO RUN
python valid.py <input_graph_file_name_list_path> <input-batches-file-path> <output_best_graph_file_name>

Scripts used:
valid_test_predict_helper.py
'''
import argparse
import pickle
import tensorflow as tf

from evaluate-v1.1 import evaluate


def validate(dataset_file, token_file, batches_file, graph_path_list_file):
    best_exact_match = -1
    best_f1 = -1
    best_graph_path = ""
    with open(graph_path_list_file , 'rb') as f:
        graph_path_list = pickle.load(f)
    with open(dataset_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    for trained_graph in graph_path_list:
        predictions = get_json_predictions(batches_file, token_file, trained_graph)
        score = evaluate(dataset, predictions)#{'exact_match': exact_match, 'f1': f1}
        #TODO: compare and update in more smart way
        if score["f1"] > best_f1:
            best_f1 = score["f1"]
            best_exact_match = score["exact_match"]
            best_graph_path = trained_graph
    return best_graph_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validating tensorflow graph')
    parser.add_argument('dataset_file')
    parser.add_argument('token_file')
    parser.add_argument('batches_file')
    parser.add_argument('graph_path_list_file')
    parser.add_argument('best_graph_path_file')
    parser.add_argument('equipment')
    args = parser.parse_args()

    #TODO: config equipment

    best_graph_path = validate(args.dataset_file, args.token_file, args.batches_file, args.graph_path_list_file )
    with open(args.best_graph_path_file, 'w') as f:
        pickle.dump(best_graph_path, f, pickle.HIGHEST_PROTOCOL)
