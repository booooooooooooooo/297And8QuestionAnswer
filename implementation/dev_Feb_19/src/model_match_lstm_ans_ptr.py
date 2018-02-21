import tensorflow as tf
import os
from tqdm import tqdm
import datetime
import random
from evaluate_v_1_1 import *
import numpy as np


'''
TODO:

use yield or dataset obj ect to read data from disk by need

write MatchLSTMCell

avoiding zero entry in softmax might cause problem in getting loss

exponential decay learning rate

mark public and private def

give nice names to important tensors, include embed_size, batch_size, num_units, lr, n_epoch etc. in title of saved model

manage scope

use more abstract ood design instead of using a lot of functions?
'''


def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

class matchLSTMcell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, h_question, question_m):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question
        self.question_m = tf.cast(question_m, tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            num_example = tf.shape(self.h_question)[0]

            init = tf.contrib.layers.xavier_initializer()

            #attention
            W_q = tf.get_variable('W_q', [self.input_size, self.input_size], dtype= tf.float32,
                                  init
                                  )
            W_c = tf.get_variable('W_c', [self.input_size, self.input_size], dtype= tf.float32,
                                  init
                                  )
            W_r = tf.get_variable('W_r', [self._state_size, self.input_size], dtype= tf.float32,
                                  init
                                  )
            W_a = tf.get_variable('W_a', [self.input_size, 1], dtype= tf.float32,
                                  init
                                  )
            b_g = tf.get_variable('b_g', [self.input_size], dtype= tf.float32,
                                  init)
            b_a = tf.get_variable('b_a', [1], dtype= tf.float32,
                                  init)


            wq_e = tf.tile(tf.expand_dims(W_q, axis=[0]), [num_example, 1, 1])
            g = tf.tanh(tf.matmul(self.h_question, wq_e)  # b x q x 2n
                        + tf.expand_dims(tf.matmul(inputs, W_c)
                                         + tf.matmul(state, W_r) + b_g, axis=[1]))


            wa_e = tf.tile(tf.expand_dims(W_a, axis=0), [num_example, 1, 1])

            a = tf.nn.softmax(tf.squeeze(tf.matmul(g, wa_e) + b_a, axis=[2]))

            a = tf.multiply(a, self.question_m)
            question_attend = tf.reduce_sum(tf.multiply(self.h_question, tf.expand_dims(a, axis=[2])), axis=1)

            z = tf.concat([inputs, question_attend], axis=1)


            W_f = tf.get_variable('W_f', (self._state_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            U_f = tf.get_variable('U_f', (2 * self.input_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            # initialize b_f with constant 1.0
            b_f = tf.get_variable('b_f', (self._state_size,), dtype = tf.float32,
                                  init)
            W_z = tf.get_variable('W_z', (self.state_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            U_z = tf.get_variable('U_z', (2 * self.input_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            # initialize b_z with constant 1.0
            b_z = tf.get_variable('b_z', (self.state_size,), dtype = tf.float32,
                                 init)  # tf.zeros_initializer())
            W_o = tf.get_variable('W_o', (self.state_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            U_o = tf.get_variable('U_o', (2 * self.input_size, self._state_size), dtype = tf.float32,
                                  init
                                  )
            b_o = tf.get_variable('b_o', (self._state_size,), dtype = tf.float32,
                                  init)

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z)
                                + tf.matmul(state, W_z) + b_z)
            f_t = tf.nn.sigmoid(tf.matmul(z, U_f)
                                + tf.matmul(state, W_f) + b_f)
            o_t = tf.nn.tanh(tf.matmul(z, U_o)
                             + tf.matmul(f_t * state, W_o) + b_o)

            output = z_t * state + (1 - z_t) * o_t
            new_state = output

        return output, new_state



class Encoder:
    def __init__(self, vocab_dim=cfg.embed_size, size=2 * cfg.lstm_num_hidden):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, context, context_m, question, question_m, embedding, keep_prob):
        context_embed = tf.nn.embedding_lookup(embedding, context)
        question_embed = tf.nn.embedding_lookup(embedding, question)

        with tf.variable_scope("encode_preprocess"):
            lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(cfg.lstm_num_hidden)
            lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(cfg.lstm_num_hidden)

            H_p_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, context_embed,
                                       sequence_length = sequence_length(context_m),
                                       dtype=tf.float32)
            H_q_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques_embed,
                                       sequence_length = sequence_length(question_m),
                                       dtype=tf.float32)
            H_p = tf.concat(H_p_pair, 2)
            H_q = tf.concat(H_q_pair, 2)


        with tf.variable_scope("encode_match"):
            lstm_gru_fw =  matchLSTMcell(2 * cfg.lstm_num_hidden, self.size, H_question,
                                              question_m)
            lstm_gru_bw = matchLSTMcell(2 * cfg.lstm_num_hidden, self.size, H_question,
                                              question_m)


            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                     lstm_gru_bw,
                                                     H_context,
                                                     sequence_length=sequence_length(context_m),
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)

        return H_r

class Decoder(object):
    def __init__(self, output_size=2 * cfg.lstm_num_hidden):
        self.output_size = output_size

    def decode(self, H_r, context_m, keep_prob):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        context_m = tf.cast(context_m, tf.float32)
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.uniform_unit_scaling_initializer(1.0)

        shape_Hr = tf.shape(H_r)
        Wr = tf.get_variable('Wr', [4 * cfg.lstm_num_hidden, 2 * cfg.lstm_num_hidden], dtype = tf.float32,
                             initializer
                             )
        Wh = tf.get_variable('Wh', [4 * cfg.lstm_num_hidden, 2 * cfg.lstm_num_hidden], dtype = tf.float32,
                             initializer
                             )
        Wf = tf.get_variable('Wf', [2 * cfg.lstm_num_hidden, 1], dtype = tf.float32,
                             initializer
                             )
        br = tf.get_variable('br', [2 * cfg.lstm_num_hidden], dtype = tf.float32,
                             initializer)
        bf = tf.get_variable('bf', [1, ], dtype = tf.float32,
                             initializer)

        wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [shape_Hr[0], 1, 1])
        f = tf.tanh(tf.matmul(H_r, wr_e) + br)


        wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [shape_Hr[0], 1, 1])
        # scores of start token.
        with tf.name_scope('starter_score'):
            s_score = tf.squeeze(tf.matmul(f, wf_e) + bf, axis=[2])
            # s_score = softmax_mask_prepro(s_score, context_m)
            # variable_summaries(s_score)
        # for checking out the probabilities of starter index
        with tf.name_scope('starter_prob'):
            s_prob = tf.nn.softmax(s_score)
            s_prob = tf.multiply(s_prob, context_m)
            # variable_summaries(s_prob)

        # logging.info('shape of s_score is {}'.format(s_score.shape))
        Hr_attend = tf.reduce_sum(tf.multiply(H_r, tf.expand_dims(s_prob, axis=[2])), axis=1)
        e_f = tf.tanh(tf.matmul(H_r, wr_e) +
                      tf.expand_dims(tf.matmul(Hr_attend, Wh), axis=[1])
                      + br)

        with tf.name_scope('end_score'):
            e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])
            # e_score = softmax_mask_prepro(e_score, context_m)
            # variable_summaries(e_score)
        # for checking out the probabilities of end index
        with tf.name_scope('end_prob'):
            e_prob = tf.nn.softmax(e_score)
            e_prob = tf.multiply(e_prob, context_m)
            # variable_summaries(e_prob)
        # logging.info('shape of e_score is {}'.format(e_score.shape))
        #TODO: why not returen prob?
        return s_score, e_score
