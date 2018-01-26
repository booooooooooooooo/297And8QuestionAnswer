

#
# with open("/output/a.txt", "w") as f:
#     f.write("hello")
# with open("/output/a.txt") as f:
#     word = f.readline()
# print word

import json
from evaluate_v_1_1 import evaluate

with open("../data/data_raw/dev-v1.1.json") as f:
    data_json = json.load(f)
    dataset = data_json['data']
with open("../data/data_raw/squad-dev-evaluate-in1") as f:
    predictions = json.load(f)

score = evaluate(dataset, predictions)
print score
