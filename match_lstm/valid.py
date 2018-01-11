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


    for v in tf.global_variables():
        print v
        print type(v)
        print sess.run(v)
        break
