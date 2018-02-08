#!/bin/bash

pass_max_length=199
ques_max_length=37
batch_size=17
embed_size=50

######Download raw data#######
#TODO#

######Split raw data to train.json, valid.json, test.json #######
python preprocess.py "get_all_json" "../data/data_raw/train-v1.1.json" "../data/data_raw/dev-v1.1.json" 0.9 "../data/data_json/"




#####Tokenize train.json, valid.json and test.json######
python preprocess.py "get_token_with_answers" "../data/data_json/train.json" "../data/data_clean/" "train" $pass_max_length
python preprocess.py "get_token_without_answers" "../data/data_json/valid.json" "../data/data_clean/" "valid" $pass_max_length

######Make voc, rev_voc using train and valid tokens  (!!DO NOT use test tokens)#######
#TODO#

######Make embedding matrix using train and valid tokens (!!DO NOT use test tokens)#####
#TODO#

######Make train ids, valid ids and test ids#######
#TODO#

######Make masked train ids, valid ids and test ids#######
#TODO#
