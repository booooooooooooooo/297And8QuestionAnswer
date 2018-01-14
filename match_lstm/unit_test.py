import tensorflow as tf
import numpy as np
import argparse
import nltk

def test_preprocess():
    from util import get_batches

    # train_batches = get_batches("./data_feed_ready/train.batches")
    # print train_batches[0]
    # valid_batches = get_batches("./data_feed_ready/valid.batches")
    # print valid_batches[0]
    # test_batches = get_batches("./data_feed_ready/test.batches")
    # print test_batches[0]

def test_helper():
    from helper import get_json_predictions

    batches_file = "./data_feed_ready/valid.batches"
    passage_tokens_file = "./data_token/valid.passage"
    question_ids_file = "./data_token/valid.question_id"
    trained_graph = "./tf_graph/10.0721421838_adam_10.0_0_2018-01-13 22:29:53.740843"
    pred_dic = get_json_predictions(batches_file, passage_tokens_file, question_ids_file, trained_graph)
    # print pred_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unit testing')
    parser.add_argument('function_name')
    args = parser.parse_args()

    if args.function_name == "helper":
        test_helper()
    elif args.function_name == "preprocess":
        test_preprocess()
