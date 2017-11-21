'''
May be unnecessary
'''

import os
import json

def split_train_to_train_and_valid(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    print data['data'][0].keys()
    print data['data'][0]['title']
    print data['data'][0]['paragraphs'][1].keys()


if __name__ == '__main__':
    split_train_to_train_and_valid("download/train-v1.1.json")
    #
    # list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    # for topic in list_topics:
    #     print topic.encode('utf-8')
