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

value error when computing softmax on zero vector,

why there is zero entry in elements. the zero entry causes log to produce nan

check initialization

compute loss more efficiently

efficient training: nce etc.

mark public and private def
give nice names to important tensors, include embed_size, batch_size, num_units, lr, n_epoch etc. in title of saved model

whether put validation and testing fuction into Model. aka how to restore graph

whether remove train part from Model. whether assing optimizer etc. in __init__
'''

class Model:
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
    def add_placeholder(self):
        batch_size = self.batch_size
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.passage = tf.placeholder(tf.float32, shape = (batch_size, pass_max_length, embed_size), name = "passage_placeholder")
        self.passage_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "passage_sequence_length_placeholder")
        self.ques = tf.placeholder(tf.float32, shape = (batch_size, ques_max_length, embed_size), name = "question_placeholder")
        self.ques_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "question_sequence_length_placeholder")
        self.ans = tf.placeholder(tf.int32, shape = (batch_size, 2), name = "answer_span_placeholder")
    def add_variables(self):
        embed_size = self.embed_size
        num_units = self.num_units
        dropout = self.dropout

        init = tf.contrib.layers.xavier_initializer()

        #Preprocessing Layer
        with tf.variable_scope("preprocessing_layer"):
            self.lstm_pre = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units), output_keep_prob=1.0 - dropout)
        #Match-LSTM Layer
        with tf.variable_scope("match_layer"):
            with tf.variable_scope("Attention"):
                self.W_q = tf.get_variable("W_q", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_p = tf.get_variable("W_p", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_r = tf.get_variable("W_r", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_p = tf.get_variable("b_p", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.w = tf.get_variable("w", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.lstm_match = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units), output_keep_prob=1.0 - dropout)
        #Answer Pointer Layer
        with tf.variable_scope("answer_pointer_layer"):
            with tf.variable_scope("Attention"):
                self.V = tf.get_variable("V", initializer = init, shape = (num_units * 2, num_units), dtype = tf.float32)
                self.W_a = tf.get_variable("W_a", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_a = tf.get_variable("b_a", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.v = tf.get_variable("v", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.c = tf.get_variable("c", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.lstm_ans = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units), output_keep_prob=1.0 - dropout)
    def pre_layer(self):
        '''
        return:
            H_p: (batch_size, pass_max_length, num_units)
            H_q: (batch_size, ques_max_length, num_units)
        '''
        passage = self.passage
        ques = self.ques
        passage_sequence_length = self.passage_sequence_length
        ques_sequence_length = self.ques_sequence_length

        lstm_pre = self.lstm_pre
        batch_size = self.batch_size
        initial_state = lstm_pre.zero_state(batch_size, dtype=tf.float32)

        #TODO：add dropout to embeddings or not?

        with tf.variable_scope("preprocessing_layer"):
            H_p, _ = tf.nn.dynamic_rnn(lstm_pre, passage, sequence_length = passage_sequence_length,
                                               initial_state=initial_state,
                                               dtype=tf.float32)
            H_q, _ = tf.nn.dynamic_rnn(lstm_pre, ques, sequence_length = ques_sequence_length,
                                               initial_state=initial_state,
                                               dtype=tf.float32)
        return H_p, H_q

    def match_attention(self, H_q, h_p, h_r):
        '''
        paras:
            H_q: (batch_size, ques_max_length, num_units)
            h_p: (batch_size, num_units)
            h_r: (batch_size, num_units)
        return:
            z: (batch_size, 2 * num_units)
        '''
        batch_size = self.batch_size
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size
        num_units = self.num_units

        ques_sequence_length = self.ques_sequence_length

        W_q = self.W_q
        W_p = self.W_p
        W_r = self.W_r
        b_p = self.b_p
        w = self.w
        b = self.b

        #TODO: use tf.expand_dims to calcualte

        #get G
        G_1 = tf.matmul( tf.reshape(H_q, (-1, num_units)), W_q)#(batch_size * ques_max_length, num_units)
        G_1 = tf.reshape(G_1, (batch_size, -1, num_units))#(batch_size, ques_max_length, num_units)
        G_2 = tf.matmul(h_p, W_p) + tf.matmul(h_r, W_r) + b_p #(batch_size, num_units)
        G_2 = tf.reshape(G_2, (batch_size, 1, num_units))#(batch_size, 1, num_units)
        G_2 = tf.tile(G_2, (1, ques_max_length, 1))#(batch_size, ques_max_length, num_units)
        G = tf.nn.tanh( G_1 + G_2 )#(batch_size, ques_max_length, num_units)
        #get alpha
        G = tf.reshape(G, (-1, num_units))#(batch_size * ques_max_length, num_units)
        alpha = tf.matmul(G, tf.reshape(w, (-1, 1))) + b #(batch_size * ques_max_length, 1)
        alpha = tf.reshape(alpha, (batch_size, -1))#(batch_size, ques_max_length)
        ques_sequence_mask = tf.sequence_mask(ques_sequence_length,ques_max_length, dtype=tf.int32)#(batch_size, ques_max_length)
        alpha += 1.0 / (ques_max_length * 100) # avoid zero vector
        alpha = alpha * tf.cast(ques_sequence_mask, tf.float32)#(batch_size, ques_max_length)
        alpha = tf.nn.softmax(alpha)#(batch_size, ques_max_length)
        #get att_encoding
        alpha = tf.reshape(alpha, (batch_size, 1, ques_max_length))#(batch_size, 1, ques_max_length)
        att_encoding = tf.matmul(alpha, H_q)#(batch_size, 1, num_units)
        att_encoding = tf.reshape(att_encoding, (batch_size, num_units))#(batch_size, num_units)
        #get z
        z = tf.concat([h_p, att_encoding], 1)#(batch_size, 2 * num_units)

        return z

    def match_one_direct(self, H_p, H_q, direction):
        '''
        paras
            H_p:        (batch_size, pass_max_length, num_units)
            H_q:        (batch_size, ques_max_length, num_units)
            direction:  "right" or "left"
        return
            H_r:        (batch_size, pass_max_length, num_units)
        '''

        num_units = self.num_units
        batch_size = self.batch_size
        pass_max_length = self.pass_max_length


        h_p_lst = tf.unstack(H_p, axis = 1)

        lstm_match = self.lstm_match
        stateTuple = lstm_match.zero_state(batch_size, tf.float32)

        h_r_lst = []
        #use name scope to avoid "kernel already exists, disallowed" error
        with tf.variable_scope("match_ayer"):
              with tf.variable_scope("LSTM"):
                if direction == "right":
                    for i in tqdm(xrange(pass_max_length), desc = "Buding matching layer on right direction"):
                        h_p = h_p_lst[i]
                        z_i = self.match_attention(H_q, h_p, stateTuple[1])
                        _, stateTuple = lstm_match(z_i, stateTuple)
                        h_r_lst.append(stateTuple[1])
                elif direction == "left":
                    for i in tqdm(range(pass_max_length - 1, -1, -1), desc = "Buding matching layer on left direction"):
                        h_p = h_p_lst[i]
                        z_i = self.match_attention(H_q, h_p, stateTuple[1])
                        _, stateTuple = lstm_match(z_i, stateTuple)
                        h_r_lst.append(stateTuple[1])
                    h_r_lst = h_r_lst[::-1]
                else:
                    raise ValueError('direction must be right or left')
        H_r = tf.stack(h_r_lst)# (pass_max_length, batch_size, num_units)
        H_r = tf.transpose(H_r, (1, 0, 2))# (batch_size, pass_max_length, num_units)
        return H_r
    def match_layer(self, H_p, H_q):
        '''
        paras
            H_p:        (batch_size, pass_max_length, num_units)
            H_q:        (batch_size, ques_max_length, num_units)
        return
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        '''
        num_units = self.num_units
        passage_sequence_length = self.passage_sequence_length#(batch_size,)
        pass_max_length = self.pass_max_length
        batch_size = self.batch_size

        #TODO: make match_lstm cell and use tf.nn.dynamic_rnn to get H_r or not?

        #get attention encoding or both directions
        H_r_right = self.match_one_direct(H_p, H_q, "right")#(batch_size, pass_max_length, num_units)
        H_r_left = self.match_one_direct(H_p, H_q, "left")#(batch_size, pass_max_length, num_units)
        #get mask
        passage_sequence_mask = tf.sequence_mask(passage_sequence_length, pass_max_length, dtype=tf.int32)#(batch_size, pass_max_length)
        passage_encoding_mask = tf.reshape(passage_sequence_mask, (batch_size, pass_max_length, 1))#(batch_size, pass_max_length, 1)
        passage_encoding_mask = tf.tile(passage_encoding_mask, (1,1, num_units))#(batch_size, pass_max_length, num_units)
        #masking
        H_r_right = H_r_right * tf.cast(passage_encoding_mask, tf.float32)
        H_r_left = H_r_left * tf.cast(passage_encoding_mask, tf.float32)
        #concat
        H_r = tf.concat([H_r_right, H_r_left], 2)#(batch_size, pass_max_length, 2 * num_units)

        return H_r

    def answer_attention(self, H_r, h_a):
        '''
        paras:
            H_r: (batch_size, pass_max_length, 2 * num_units)
            h_a: (batch_size, num_units)
        return:
            beta: (batch_size, pass_max_length)
            att_encoding: (batch_size, 2 * num_units)

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

        #TODO: use tf.expand_dims to calculate

        #get F
        F_1 = tf.matmul(tf.reshape(H_r, (-1, 2 * num_units)), V)#(batch_size * pass_max_length, num_units)
        F_1 = tf.reshape(F_1, (batch_size, pass_max_length, num_units))#(batch_size, pass_max_length, num_units)
        F_2 = tf.matmul(h_a, W_a) + b_a # (batch_size, num_units)
        F_2 = tf.reshape(F_2, (batch_size, 1, num_units))# (batch_size, 1, num_units)
        F_2 = tf.tile(F_2, (1, pass_max_length, 1))#(batch_size, pass_max_length, num_units)
        F = tf.nn.tanh(F_1 + F_2)#(batch_size, pass_max_length, num_units)
        #get beta
        F = tf.reshape(F, (-1, num_units))#(batch_size * pass_max_length, num_units)
        beta = tf.matmul(F, tf.reshape(v, (-1, 1))) + c #(batch_size * pass_max_length, 1)
        beta = tf.reshape(beta, (batch_size, pass_max_length))#(batch_size, pass_max_length)
        pass_sequence_mask = tf.sequence_mask(passage_sequence_length,pass_max_length, dtype=tf.int32)#(batch_size, pass_max_length)
        beta += 1.0 / (pass_max_length * 100) # avoid zero vector
        beta = beta * tf.cast(pass_sequence_mask, tf.float32)#(batch_size, pass_max_length)
        beta = tf.nn.softmax(beta)#(batch_size, pass_max_length)
        #get att_encoding
        att_encoding = tf.matmul( tf.reshape(beta, (batch_size, 1, pass_max_length)), H_r )#(batch_size, 1, 2 * num_units)
        att_encoding = tf.reshape(att_encoding, (batch_size, 2 * num_units))

        return beta, att_encoding

    def answer_layer(self, H_r):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''
        num_units = self.num_units
        batch_size = self.batch_size
        lstm_ans = self.lstm_ans

        stateTuple = lstm_ans.zero_state(batch_size, tf.float32)

        dist_lst = []
        with tf.variable_scope("answer_pointer_layer"):
              with tf.variable_scope("LSTM"):
                for i in tqdm(xrange(2), desc = "Building answer pointer layer") :
                    beta, att_encoding = self.answer_attention(H_r, stateTuple[1])
                    _, stateTuple = lstm_ans(att_encoding, stateTuple)
                    dist_lst.append(beta)
        dist = tf.stack(dist_lst)#(2, batch_size, pass_max_length)
        dist = tf.transpose(dist, (1,0,2), name='dist')#(batch_size, 2, pass_max_length)
        return dist

    def add_predicted_dist(self):
        #Preprocessing Layer
        H_p, H_q = self.pre_layer()
        #Match-LSTM Layer
        H_r = self.match_layer(H_p, H_q)
        #Answer Pointer Layer
        dist = self.answer_layer(H_r)

        self.dist = dist
    def add_loss_function(self):
        pass_max_length = self.pass_max_length
        batch_size = self.batch_size

        ans = self.ans#(batch_size, 2)
        dist = self.dist#(batch_size, 2, pass_max_length)

        mask = tf.one_hot(ans, pass_max_length, dtype=tf.bool, on_value=True, off_value=False)#(batch_size, 2, pass_max_length)
        elements = tf.boolean_mask(dist, mask)#(?,)
        loss =  -tf.reduce_mean( tf.log( elements ) )

        self.loss = loss
    def build(self):
        #add placeholders
        self.add_placeholder()
        #add variables / make architectures
        self.add_variables()
        #add predicted distribution
        self.add_predicted_dist()
        #add loss
        self.add_loss_function()
