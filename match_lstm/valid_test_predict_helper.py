
def get_json_predictions(input_json_data, trained_graph):
    return json_predictions
def get_score(input_json_data, trained_graph):
    json_predictions = get_json_predictions(input_json_data, trained_graph)
    #evaluate json_predictions using evaluate-1.1.py
    return score
