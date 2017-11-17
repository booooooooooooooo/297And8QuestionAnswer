import os
import json
# class Match_lstm_odm:
#     #TODO
if __name__ == '__main__':
    with open("download/train-v1.1.json") as data_file:
        data = json.load(data_file)
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    for topic in list_topics:
        print topic.encode('utf-8')
