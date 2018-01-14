'''
Used for online submission.

TO RUN:
python src/<path-to-prediction-program> <input-data-json-file> <output-prediction-json-path>


'''

import tensorflow as tf
import argparse
import os

from preprocess import Preprocessor
from helper import *

def predict(json_file, prediction_json_path):
    #manually updated
    trained_graph = "./tf_graph/10.0721421838_adam_10.0_0_2018-01-13 22:29:53.740843"
    pass_max_length=13
    ques_max_length=7
    batch_size=97
    embed_size=50
    glove_file = "./data_raw/glove.6B/glove.6B.50d.txt"
    dir_to_save = "./predict/"
    prefix = "predict"

    #get passage_tokens_file, question_ids_file
    preprocessor = Preprocessor()
    preprocessor.get_token_without_answers(json_file, dir_to_save, prefix)
    #batches_file
    passage_tokens_file = os.path.join(dir_to_save, prefix + ".passage")
    question_tokens_file = os.path.join(dir_to_save, prefix + ".question")
    question_ids_file = os.path.join(dir_to_save, prefix + ".question_id")
    preprocessor.get_batches_without_answers(pass_max_length, ques_max_length, batch_size, embed_size, passage_tokens_file, question_tokens_file, question_ids_file, glove_file, dir_to_save, prefix)
    #get predictions
    batches_file = os.path.join(dir_to_save, prefix + ".batches")
    predictions = get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph)
    #write predictions
    with open(prediction_json_path, 'w') as f:
        f.write(json.dumps(predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validating tensorflow graph')
    parser.add_argument('json_file')
    parser.add_argument('prediction_json_path')
    args = parser.parse_args()

    predict(args.json_file, args.prediction_json_path)
