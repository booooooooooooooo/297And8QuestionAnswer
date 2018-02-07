

#
# with open("/output/a.txt", "w") as f:
#     f.write("hello")
# with open("/output/a.txt") as f:
#     word = f.readline()
# print word

# import json
# from evaluate_v_1_1 import evaluate
#
# with open("../data/data_raw/dev-v1.1.json") as f:
#     data_json = json.load(f)
#     dataset = data_json['data']
# with open("../data/data_raw/squad-dev-evaluate-in1") as f:
#     predictions = json.load(f)
#
# score = evaluate(dataset, predictions)
# print score

# import os
# print os.path.dirname(__file__)

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"

glove_dic = {}
with open("../data/data_raw/glove.6B/glove.6B.50d.txt") as fh:
    for line in fh:
        line_list = line.split()
        word = line_list[0].decode('utf8')#decode
        vector = line_list[1:]
        glove_dic[word] = vector
print glove_dic[_UNK]
