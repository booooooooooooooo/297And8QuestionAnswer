import tensorflow as tf
import numpy as np

def sanity_model():
    from model import Model
    class Config:
        def __init__(self, batch_s, embed_s, num_units, embed_matrix, n_epoch):
            self.batch_s = batch_s
            self.embed_s = embed_s
            self.num_units = num_units
            self.embed_matrix = embed_matrix
            self.n_epoch = n_epoch
    class Data:
        def getTrain(self, embed_s, batch_s):
            pass_l = 119
            ques_l = 13
            vocabulary = None#TODO
            vocabulary_rev = None#TODO
            embed_matrix = tf.ones((17, 17))
            batch_data = None#TODO
            return pass_l, ques_l, vocabulary, vocabulary_rev, embed_matrix, batch_data


    batch_s = 2
    embed_s = 3
    num_units = 5
    n_epoch = 7
    data = Data()
    pass_l, ques_l, vocabulary, vocabulary_rev, embed_matrix, batch_data = data.getTrain(embed_s, batch_s)
    config = Config(batch_s, embed_s, num_units, embed_matrix, n_epoch)


    model = Model(config)

    model.add_placeholder()
    print model.ques
    print model.ques_mask

    model.add_variables()
    print model.W_q

    pass_embed, ques_embed = model.embed_layer()
    print pass_embed
    print ques_embed

    H_p, H_q = model.pre_layer(pass_embed, ques_embed)
    print H_p
    print H_q

def test_tensorflow():
    ##test BasicRNNCell and dynamic_rnn
    batch_s = 7
    max_length = 5
    input_size = 3
    hidden_size = 2
    input_data = tf.ones((batch_s, max_length, input_size))
    input_data_seq_length = tf.ones((batch_s,))

    # create a BasicRNNCell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    # 'outputs' is a tensor of shape [batch_s, max_time, cell_state_size]

    # defining initial state
    initial_state = rnn_cell.zero_state(batch_s, dtype=tf.float32)

    # 'state' is a tensor of shape [batch_s, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data, sequence_length = input_data_seq_length,
                                       initial_state=initial_state,
                                       dtype=tf.float32)

    # input_data_prime = tf.zeros((batch_s, max_length, input_size * 3))
    # outputs_prime, state_prime = tf.nn.dynamic_rnn(rnn_cell, input_data_prime,
    #                                    initial_state=initial_state,
    #                                    dtype=tf.float32)
    # print outputs_prime
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer() )
        print sess.run(outputs)

if __name__ == "__main__":
    # test_tensorflow()
    sanity_model()
