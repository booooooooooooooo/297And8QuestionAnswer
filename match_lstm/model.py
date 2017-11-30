'''
Used by tune.py to train better tensor graph.
'''
import tensorflow as tf
import numpy as np
from util import *
'''
TODO:
padding
mark public and private def
'''
class Model:
    def __init__(self, config):
        self.config = config
    def add_placeholder(self):
        pass_l = self.config.data.pass_l
        ques_l = self.config.data.ques_l

        self.ques = tf.placeholder(tf.int32, shape = (None, ques_l), name = "ques_ph")
        self.ques_mask = tf.placeholder(tf.int32, shape = (None, ques_l), name = "ques_pad_ph")
        self.passage = tf.placeholder(tf.int32, shape = (None, pass_l), name = "pass_ph")
        self.passage_mask = tf.placeholder(tf.int32, shape = (None, pass_l), name = "pass_pad_ph")
        self.ans = tf.placeholder(tf.int32, shape = (None, 2), name = "ans_ph")
    def add_variables(self):
        embed_s = self.config.data.embed_s
        num_units = self.config.num_units

        init = tf.contrib.layers.xavier_initializer()

        #Preprocessing Layer
        with tf.variable_scope("Preprocessing Layer"):
            self.U_pre = tf.get_variable("U_pre", initializer = init, shape = (embed_s, num_units))
            self.lstm_pre = tf.contrib.rnn.BasicLSTMCell(num_units)
        #Match-LSTM Layer
        with tf.variable_scope("Match-LSTM Layer"):
            with tf.variable_scope("Attention"):
                self.W_q = tf.get_variable("W_q", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_p = tf.get_variable("W_p", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_r = tf.get_variable("W_r", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_p = tf.get_variable("b_p", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.w = tf.get_variable("w", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.U_match = tf.get_variable("U_match", initializer = init, shape = (2 * num_units, num_units))
                self.lstm_match = tf.contrib.rnn.BasicLSTMCell(num_units)
        #Answer Pointer Layer
        with tf.variable_scope("Answer Pointer Layer"):
            with tf.variable_scope("Attention"):
                self.V = tf.get_variable("V", initializer = init, shape = (num_units * 2, num_units), dtype = tf.float32)
                self.W_a = tf.get_variable("W_a", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_a = tf.get_variable("b_a", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.v = tf.get_variable("v", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.c = tf.get_variable("c", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.U_ans = tf.get_variable("U_ans", initializer = init, shape = (2 * num_units, num_units))
                self.lstm_ans = tf.contrib.rnn.BasicLSTMCell(num_units)
    def embed_layer(self):
        embed_matrix = self.config.data.embed_matrix
        passage = self.passage
        ques = self.ques

        pass_embed = tf.nn.embedding_lookup(embed_matrix, passage)#(None, pass_l, embed_s)
        ques_embed = tf.nn.embedding_lookup(embed_matrix, ques)#(None, ques_l, embed_s)
        return pass_embed, ques_embed
    def pre_layer(self, pass_embed, ques_embed):
        U_pre = self.U_pre
        pass_l = self.config.data.pass_l
        embed_s = self.config.data.embed_s
        num_units = self.config.num_units
        lstm_pre = self.lstm_pre
        passage_mask = self.passage_mask
        ques_mask = self.ques_mask

        pass_pre = abc_mul_cd(pass_embed, U_pre, pass_l, embed_s, num_units)#(None, pass_l, num_units)
        ques_pre = abc_mul_cd(ques_embed, U_pre, ques_l, embed_s, num_units)#(None, ques_l, num_units)
        H_p, _ = tf.nn.dynamic_rnn(lstm_pre, pass_pre, passage_mask, dtype=tf.float32)#(None, pass_l, num_units)
        H_q, _ = tf.nn.dynamic_rnn(lstm_pre, ques_pre, ques_mask, dtype=tf.float32)#(None, ques_l, num_units)
        return H_p, H_q
    def match_attention(self, H_q, h_p, h_r):
        '''
        paras:
            H_q: (batch_size, ques_l, num_units)
            h_p: (batch_size, num_units)
            h_r: (batch_size, num_units)
        return:
            z: (batch_size, 2 * num_units)
        '''
        ques_l = self.config.data.ques_l
        num_units = self.config.num_units
        W_q = self.W_q
        W_p = self.W_p
        W_r = self.W_r
        b_p = self.b_p
        w = self.w
        b = self.b

        G_part1 = abc_mul_cd(H_q, W_q, ques_l, num_units, num_units)#(batch_size, ques_l, num_units)
        G_part2 = tf.matmul(h_p, W_p) + tf.matmul(h_r, W_r) + b_p #(batch_size, num_units)
        G_sum = abc_plus_ac(G_part1, G_part2, ques_l, num_units)#(batch_size, ques_l, num_units)
        G = tf.nn.tanh( G_sum )# batch_size * ques_l * l

        alpha = tf.nn.softmax( abc_mul_c(G, w, ques_l, num_units) + b )#(batch_size, ques_l)

        att_v = tf.matmul( tf.transpose(H_q, (0,2,1)), tf.reshape(alpha, (-1, ques_l, 1)) )#(batch_size, num_units, 1)
        att_v = tf.reshape(att_v, (-1, num_units))#(batch_size, num_units)

        z = tf.concat(1, [h_p, att_v])#(batch_size, 2 * num_units)

        return z
    def match_layer(self, H_p, H_q):
        '''
        paras
            H_p:        (None, pass_l, num_units)
            H_q:        (None, ques_l, num_units)
        return
            H_r:        (None, pass_l, num_units)
        '''
        lstm = self.lstm_match
        num_units = self.config.num_units
        batch_size = self.config.data.batch_size

        h_p_lst = tf.unstack(H_p, axis = 1)

        h_r_lst = []
        h_r = lstm.zero_state(batch_size, tf.float32)
        for i in xrange(pass_len):
            h_p = h_p_lst[i]
            z_i = match_attention(H_q, h_p, h_r)
            h_r = lstm(z_i, h_r)
            h_r_lst.append(h_r)
        H_r = tf.stack(h_r_lst)# (pass_len, None, num_units)

        return H_r
    def answer_attention(self, H_r, h_a):
        '''
        paras:
            H_r: (batch_size, pass_l, 2 * num_units)
            h_a: (batch_size, num_units)
        return:
            beta: (batch_size, pass_l)
            input_lstm: (batch_size, 2 * num_units)

        '''
        V = self.V
        pass_l = self.config.data.pass_l
        num_units = self.config.num_units
        W_a = self.W_a
        b_a = self.b_a
        v = self.v
        c = self.c

        F_part1 = abc_mul_cd(H_r, V, pass_l, 2 * num_units, num_units)# (batch_size, pass_l, num_units)
        F_part2 = tf.matmul(h_a, W_a) + b_a # (batch_size, num_units)
        F_sum = abc_plus_ac(F_part1, F_part2, pass_l, num_units )
        F = tf.nn.tanh( F_sum )# (batch_size, pass_l, num_units)

        beta = tf.nn.softmax( abc_mul_c(F, v, pass_l, num_units) + c )#(batch_size, pass_l)

        input_lstm = tf.matmul( tf.transpose(H_r, (0,2,1)), tf.reshape(beta, (-1, pass_l, 1)) )#(batch_size, 2 * num_units, 1)
        input_lstm = tf.reshape(input_lstm, (-1, 2 * num_units))

        return beta, input_lstm
    def answer_layer(self, H_r):
        '''
        paras
            H_r:        (None, pass_l, 2 * num_units)
        return
            dist:       (None, 2, pass_l)
        '''
        lstm = self.lstm_ans
        num_units = self.config.num_units
        batch_size = self.config.data.batch_size

        dist_lst = []
        h_a = lstm.zero_state(batch_size, tf.float32)
        for i in xrange(2):
            beta, input_lstm = match_attention(H_r, h_a)
            h_a = tf.matmul(input_lstm, U_ans)
            dist_lst.append(beta)
        dist = tf.stack(dist_lst)#(2, None, pass_l)

        return tf.transpose(dist, (1,0,2))#(None, 2, pass_l)
    def add_predicted_dist(self):
        #embdding
        pass_embed, ques_embed = self.embed_layer()
        #Preprocessing Layer
        H_p, H_q = pre_layer(pass_embed, ques_embed)
        #Match-LSTM Layer
        H_r_r = match_layer(H_p, H_q)
        H_r_l = match_layer(H_p, H_q) #TODO: bidirectional lstm and padding
        H_r = tf.concat(2, [H_r_r, H_r_l]) ## (pass_len, None, 2 * l)
        H_r = tf.transpose(H_r, (1, 0, 2))
        #Answer Pointer Layer
        self.dist = answer_layer(H_p)

    def add_train_op(self):
        #TODO
        loss = tf.reduce_mean( )
    def build(self):
        #add placeholders
        add_placeholder()
        #add variables / make architectures
        add_variables()
        #get predicted distribution
        add_predicted_dist()
        #add train_op
        add_train_op()

    def train_batch(self, sess, batch_data):

    def train_epoch(self, sess):

    def fit(self, sess):
