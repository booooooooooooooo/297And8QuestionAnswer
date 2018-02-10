'''
Part of graph used by both train, valid and test.
'''
import tensorflow as tf
import os
from tqdm import tqdm
from datetime import datetime

from util import *

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

class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units):
        self._input_size = num_units
        self._state_size = 2 * num_units
    @property
    def state_size(self):
        return self._state_size
    @property
    def output_size(self):
        return self._state_size
    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            

class MatchLSTMAnsPtr:
    def add_placeholder(self):
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.passage = tf.placeholder(tf.float32, shape = (batch_size, pass_max_length, embed_size), name = "passage_placeholder")
        self.passage_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "passage_sequence_length_placeholder")
        self.ques = tf.placeholder(tf.float32, shape = (batch_size, ques_max_length, embed_size), name = "question_placeholder")
        self.ques_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "question_sequence_length_placeholder")
        self.ans = tf.placeholder(tf.int32, shape = (batch_size, 2), name = "answer_span_placeholder")
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = "keep_prob_placeholder")
    def add_variables(self):
        embed_size = self.embed_size
        num_units = self.num_units

        init = tf.contrib.layers.xavier_initializer()


        with tf.variable_scope("encode_preprocess"):
            self.lstm_pre = tf.contrib.rnn.BasicLSTMCell(num_units)

        with tf.variable_scope("encode_match"):
            #TODO
            # self.lstm_gru_fw
            # self.lstm_gru_bw

        with tf.variable_scope("decode_ans_ptr"):
            self.V = tf.get_variable("V", initializer = init, shape = (num_units * 2, num_units), dtype = tf.float32)
            self.W_a = tf.get_variable("W_a", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
            self.b_a = tf.get_variable("b_a", initializer = init, shape = (num_units, ), dtype = tf.float32)
            self.v = tf.get_variable("v", initializer = init, shape = (num_units, ), dtype = tf.float32)
            self.c = tf.get_variable("c", initializer = init, shape = (), dtype = tf.float32)

    def encode_preprocess(self, passage, passage_sequence_length, ques, ques_sequence_length, keep_prob,lstm_pre ):
        '''
        return:
            H_p: (batch_size, pass_max_length, num_units)
            H_q: (batch_size, ques_max_length, num_units)
        '''

        with tf.variable_scope("preprocessing_layer"):
            H_p, _ = tf.nn.dynamic_rnn(lstm_pre, passage,
                                       sequence_length = passage_sequence_length,
                                       dtype=tf.float32)
            H_q, _ = tf.nn.dynamic_rnn(lstm_pre, ques,
                                       sequence_length = ques_sequence_length,
                                       dtype=tf.float32)
            H_p = tf.nn.dropout(H_p, keep_prob=keep_prob)
            H_q = tf.nn.dropout(H_q, keep_prob=keep_prob)
        return H_p, H_q

    def encode_match(self, H_p, passage_sequence_length, keep_prob, lstm_gru_fw, lstm_gru_bw):
        '''
        paras
            H_p:        (batch_size, pass_max_length, num_units)
        return
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        '''
        with tf.variable_scope("encode_match"):
            H_r, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                     lstm_gru_bw,
                                                     H_p,
                                                     sequence_length=passage_sequence_length,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r, 2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)
        return H_r

    def decode_ans_ptr(self, H_r, keep_prob, V, W_a, b_a, v, c):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''
        #TODO: make this functional. Avoid use self. ??

            return dist
        '''
        def answer_attention(self, H_r, h_a):
            '''
            paras:
                H_r: (batch_size, pass_max_length, 2 * num_units)
                h_a: (batch_size, num_units)
            return:
                beta: (batch_size, pass_max_length)
                att_v: (batch_size, 2 * num_units)

            '''
            batch_size = self.batch_size
            pass_max_length = self.pass_max_length
            passage_sequence_length = self.passage_sequence_length
            num_units = self.num_units

            V = self.V#(2 * num_units, num_units)
            W_a = self.W_a#(num_units, num_units)
            b_a = self.b_a#(num_units, )
            v = self.v#(num_units, )
            c = self.c#()

            F = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                        +  tf.expand_dims(tf.matmul(h_a, W_a) + b_a , [1]))#(batch_size, pass_max_length, num_units)
            beta = tf.reshape(tf.matmul(F, tf.tile(tf.reshape(v, [1, num_units, 1]), [batch_size, 1, 1])), [batch_size, pass_max_length])#(batch_size, pass_max_length)
            pass_sequence_mask = tf.sequence_mask(passage_sequence_length,pass_max_length, dtype=tf.int32)#(batch_size, pass_max_length)
            beta += 1.0 / (pass_max_length * 100) # avoid zero vector
            beta = beta * tf.cast(pass_sequence_mask, tf.float32)#(batch_size, pass_max_length)
            beta = tf.nn.softmax(beta)#(batch_size, pass_max_length)
            att_v = tf.reshape(tf.matmul(tf.expand_dims(beta, [1]), H_r), [batch_size, 2 * num_units])#(batch_size, 2 * num_units)

            return beta, att_v
        '''

    def add_predicted_dist(self):
        #Preprocessing Layer
        H_p, H_q = self.encode_preprocess(False)
        #Match-LSTM Layer
        H_r = self.encode_match(H_p, H_q, False)
        #Answer Pointer Layer
        dist = self.decode_ans_ptr(H_r, False)
        self.dist = dist

        #Preprocessing Layer
        H_p_dropout, H_q_dropout = self.encode_preprocess(True)
        #Match-LSTM Layer
        H_r_dropout = self.encode_match(H_p, H_q, True)
        #Answer Pointer Layer
        dist_dropout = self.decode_ans_ptr(H_r, True)
        self.dist_dropout = dist_dropout
    def add_loss_function(self):
        pass_max_length = self.pass_max_length
        batch_size = self.batch_size

        ans = self.ans#(batch_size, 2)

        dist = self.dist#(batch_size, 2, pass_max_length)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=dist, labels=ans)
        self.loss = tf.reduce_mean(loss)

        dist_dropout = self.dist_dropout#(batch_size, 2, pass_max_length)
        loss_dropout = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=dist_dropout, labels=ans)
        self.loss_dropout = tf.reduce_mean(loss_dropout)
    def build(self):
        #add placeholders
        self.add_placeholder()
        #add variables / make architectures
        self.add_variables()
        #add predicted distribution
        self.add_predicted_dist()
        #add loss
        self.add_loss_function()
    def __init__(self, pass_max_length, ques_max_length, batch_size, embed_size, num_units, dropout):
        #parameters used by the graph. Train, valid and test data must be consistent on these parameters.
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length#not sure
        self.batch_size = batch_size
        self.embed_size = embed_size
        #parameter used by the graph. It is not related to data.
        self.num_units = num_units
        self.dropout = dropout
        #build the graph
        self.build()
