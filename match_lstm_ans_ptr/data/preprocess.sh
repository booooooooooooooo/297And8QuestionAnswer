#!/bin/bash

pass_max_length=766
ques_max_length=60
batch_size=97
embed_size=50

######preprocess train#######
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_with_answers" "./data_json/train.json" "./data_token/" "train" $pass_max_length
python preprocess.py "get_batches_with_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/train.passage" "./data_token/train.question" "./data_token/train.answer_span" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "train"

######preprocess valid#######
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_without_answers" "./data_json/valid.json" "./data_token/" "valid"
python preprocess.py "get_batches_without_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/valid.passage" "./data_token/valid.question" "./data_token/valid.question_id" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "valid"

######preprocess test#######
python preprocess.py "get_all_json" "./data_raw/train-v1.1.json" "./data_raw/dev-v1.1.json" 0.9 "./data_json/"
python preprocess.py "get_token_without_answers" "./data_json/test.json" "./data_token/" "test"
python preprocess.py "get_batches_without_answers" $pass_max_length $ques_max_length $batch_size $embed_size "./data_token/test.passage" "./data_token/test.question" "./data_token/test.question_id" "./data_raw/glove.6B/glove.6B.50d.txt" "./data_feed_ready/" "test"
