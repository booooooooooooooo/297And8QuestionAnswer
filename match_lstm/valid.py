'''
Used for validation.

TO RUN
python valid.py <input_graph_file_name_list_path> <input-batches-file-path> <output_best_graph_file_name>

Scripts used:
valid_test_predict_helper.py
'''
import pickle
import tensorflow as tf

with open("./output/graph_name_list", 'rb') as f:
    file_to_save_graph_name_list = pickle.load(f)
for f in file_to_save_graph_name_list:
    print f

targetFile = file_to_save_graph_name_list[0]




with tf.Session() as sess:
    saver = tf.train.import_meta_graph(targetFile + '.meta')
    saver.restore(sess, targetFile)
    print("Model restored.")


    # for v in tf.global_variables():
    #     print v
    #     print type(v)
    #     print sess.run(v)
    #     break

    passage_ph = tf.get_default_graph().get_tensor_by_name("passage_placeholder:0")
    passage_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("passage_sequence_length_placeholder:0")
    ques_ph = tf.get_default_graph().get_tensor_by_name("question_placeholder:0")
    ques_sequence_length_ph = tf.get_default_graph().get_tensor_by_name("question_sequence_length_placeholder:0")
    dist = tf.get_default_graph().get_tensor_by_name("dist:0")
    print passage_ph
    print passage_sequence_length_ph
    print ques_ph
    print ques_sequence_length_ph
    print dist

    # predicted_dist = sess.run(dist, {passage_ph : passage,
    #                                                   passage_sequence_length_ph : passage_sequence_length,
    #                                                   ques_ph : ques,
    #                                                   ques_sequence_length_ph : ques_sequence_length)
