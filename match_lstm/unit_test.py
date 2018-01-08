import tensorflow as tf
import numpy as np
import argparse
import nltk


def test_tensorflow():
    # ##test BasicRNNCell and dynamic_rnn
    # batch_s = 7
    # max_length = 5
    # input_size = 3
    # hidden_size = 2
    # input_data = tf.ones((batch_s, max_length, input_size))
    # input_data_seq_length = tf.ones((batch_s,))
    #
    # # create a BasicRNNCell
    # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    #
    # # 'outputs' is a tensor of shape [batch_s, max_time, cell_state_size]
    #
    # # defining initial state
    # initial_state = rnn_cell.zero_state(batch_s, dtype=tf.float32)
    #
    # # 'state' is a tensor of shape [batch_s, cell_state_size]
    # outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data, sequence_length = input_data_seq_length,
    #                                    initial_state=initial_state,
    #                                    dtype=tf.float32)
    #
    # # input_data_prime = tf.zeros((batch_s, max_length, input_size * 3))
    # # outputs_prime, state_prime = tf.nn.dynamic_rnn(rnn_cell, input_data_prime,
    # #                                    initial_state=initial_state,
    # #                                    dtype=tf.float32)
    # # print outputs_prime
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer() )
    #     print sess.run(outputs)



    # ##test reshape
    # a = tf.constant([[[1,2,3], [4,5,6]], [[7,8,9], [10, 11, 12]]])
    # b = tf.transpose(a, (0,2,1))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer() )
    #     print sess.run(a)
    #     print sess.run(b)
    #     print sess.run(tf.reshape(b, (-1, 2)  ))

    # ##test tf.matmul
    # a = tf.ones((3, 4))
    # b = tf.ones((4,1))
    # c = tf.matmul(a, b)
    # print c


    # ##test while_loop
    # i0 = tf.constant(0)
    # m0 = tf.ones([2, 2])
    # c = lambda i, m: i < 10
    # b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
    # loop = tf.while_loop(
    #     c, b, loop_vars=[i0, m0],
    #     shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer() )
    #     print sess.run(loop)

    # ##test case
    # x = 1
    # y = 2
    #
    # f1 = lambda: tf.constant(17)
    # f2 = lambda: tf.constant(23)
    # r = tf.case([(tf.less(x, y), f1)], default=f2)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer() )
    #     print sess.run(r)

    ##test unicode
    a = unicode("how are you")
    print a
    for b in enumerate(a):
        print b,

def test_preprocess():
    from preprocess import Preprocessor
    pass_ques_ans_json_path = "./data_raw/fake_pass_ques_ans.json"
    dir_to_save = "./data_token/"
    train_percent = 0.9

    preprocessor = Preprocessor()

    preprocessor.preprocess_train_json_to_train_and_valid_token(pass_ques_ans_json_path, dir_to_save, train_percent)

def test_tokenizer():
    with open("./output/failedPreprocessCase") as fh:
        context = fh.readline()
    context = context.decode('utf8')
    context_token = nltk.word_tokenize(context)
    context_token = [token.replace("``", '"').replace("''", '"') for token in context_token]#nltk makes "aaa "word" bbb" to 'aaa', '``', 'word', '''', 'bbb'

    print type(context)
    print context.encode('utf8')
    print len(context)
    for token in context_token:
        print token.encode('utf8'),
    print
    print len(context_token)

    c_id_to_token_id_map = {}
    token_id = 0
    id_in_cur_token = 0
    for c_id, c in enumerate(context):
        if nltk.word_tokenize(c) != []:
            if id_in_cur_token == len(context_token[token_id]):
                token_id += 1
                id_in_cur_token = 0
            c_id_to_token_id_map[c_id] = token_id
            print c_id
            print c.encode('utf8')
            print id_in_cur_token
            print context_token[token_id].encode('utf8')
            print "======================="
            id_in_cur_token += 1
def test_midprocess():
    from midprocess import Midprocessor

    midprocessor = Midprocessor()


    passage_file = "./data_token/train.passage"
    question_file ="./data_token/train.question"
    answer_span_file ="./data_token/train.answer_span"
    glove_path = "./data_raw/glove.6B/glove.6B.50d.txt"
    batches_file = "./data_feed_ready/train.batches"

    midprocessor.get_padded_vectorized_and_batched(passage_file, question_file, answer_span_file, glove_path, batches_file)
def test_train():
    from train import get_batches
    batches_file = "./data_feed_ready/train.batches"
    batches = get_batches(batches_file)
    print len(batches)
    print len(batches[0])
    print len(batches[0][0])
    print len(batches[0][1])
    print len(batches[0][2])

    print len(batches[0][0][0])
    print len(batches[0][1][0])
    print len(batches[0][2][0])

def test_model():
    import tensorflow as tf
    from model import Model


    pass_max_length = 9
    ques_max_length = 8
    batch_size = 7
    embed_size = 6

    num_units = 5
    optimizer = "adam"
    lr = 0.5
    n_epoch=5
    batches_file="./data_feed_ready/train.batches"
    dirToSaveModel="./tf_graph/"



    my_model = Model(pass_max_length, ques_max_length, batch_size, embed_size, num_units)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    saved_model_list =  my_model.fit(sess, optimizer, lr, n_epoch, batches_file, dirToSaveModel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unit testing')
    parser.add_argument('filename')
    args = parser.parse_args()
    if args.filename == "preprocess":
        test_preprocess()
    elif args.filename == "tokenizer":
        test_tokenizer()
    elif args.filename == "midprocess":
        test_midprocess()
    elif args.filename == "train":
        test_train()
    elif args.filename == "model":
        test_model()
