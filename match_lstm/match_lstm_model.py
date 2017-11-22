import tensorflow as tf
import numpy as np
from match_lstm_model_helper import *

class LSTM_encoder:
    def __init__(self, scopeName, input_size, state_size):
        self.scopeName = scopeName
        self.input_size = input_size
        self.state_size = state_size
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scopeName):
            self.W_i = tf.get_variable("W_i", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_i = tf.get_variable("V_i", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_i = tf.get_variable("b_i", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_f = tf.get_variable("W_f", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_f = tf.get_variable("V_f", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_f = tf.get_variable("b_f", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_o = tf.get_variable("W_o", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_o = tf.get_variable("V_o", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_o = tf.get_variable("b_o", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_c = tf.get_variable("W_c", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_c = tf.get_variable("V_c", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_c = tf.get_variable("b_c", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)
    def encode_one_step(self, batch_inputs, batch_states, batch_memories):
        i_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_i) + tf.matmul(batch_states, self.V_i) + self.b_i)
        f_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_f) + tf.matmul(batch_states, self.V_f) + self.b_f)
        o_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_o) + tf.matmul(batch_states, self.V_o) + self.b_o)
        new_batch_memories = f_k * batch_memories + i_k * tf.nn.tanh( tf.matmul(batch_inputs, self.W_c) + tf.matmul(batch_states, self.V_c) + self.b_c)
        new_batch_states = o_k * tf.nn.tanh(new_batch_memories)

        return new_batch_states, new_batch_memories
    def encode_sequence(self, batch_inputs_sequence, seq_len):
        '''
        paras
            batch_inputs_sequence: (None, seq_len, input_size)
        return
            encoded: (None, seq_len, state_size)
        '''
        # Assume all sequences have same length. Padding work is done in data related files.
        states = []
        batch_states = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        batch_memories = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        for i in xrange(seq_len):
            new_batch_states, new_batch_memories = self.encode_one_step(batch_inputs_sequence[:,i], batch_states, batch_memories)
            states.append(new_batch_states)
            batch_states, batch_memories = new_batch_states, new_batch_memories
        encoded = tf.transpose( tf.stack(states) , (1, 0, 2))
        return encoded

class Attention_match:
    def __init__(self, scopeName, l):
        self.l = l
        self.scopeName = scopeName
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scopeName):
            self.W_q = tf.get_variable("W_q", initializer = xavier_initializer, shape = (self.l, self.l), dtype = tf.float32)
            self.W_p = tf.get_variable("W_p", initializer = xavier_initializer, shape = (self.l, self.l), dtype = tf.float32)
            self.W_r = tf.get_variable("W_r", initializer = xavier_initializer, shape = (self.l, self.l), dtype = tf.float32)
            self.b_p = tf.get_variable("b_p", initializer = xavier_initializer, shape = (self.l, ), dtype = tf.float32)
            self.w = tf.get_variable("w", initializer = xavier_initializer, shape = (self.l, ), dtype = tf.float32)
            self.b = tf.get_variable("b", initializer = xavier_initializer, shape = (), dtype = tf.float32)
    def attention_one_step(self, H_q, q_l, h_p, h_r):
        '''
        paras:
            H_q: batch_size * q_l * l
            h_p: batch_size * l
            h_r: batch_size * l
        return:
            z: batch_size * (2 * l)
        '''

        G_part1 = abc_mul_cd(H_q, self.W_q, q_l, self.l, self.l)# batch_size * q_l * l
        G_part2 = tf.matmul(h_p, self.W_p) + tf.matmul(h_r, self.W_r) + self.b_p # batch_size * l
        G_sum = abc_plus_ac(G_part1, G_part2, q_l, self.l)
        G = tf.nn.tanh( G_sum )# batch_size * q_l * l

        alpha = tf.nn.softmax( abc_mul_c(G, self.w, q_l, self.l) + self.b )# batch_size * q_l

        att_v = tf.matmul( tf.transpose(H_q, (0,2,1)), tf.reshape(alpha, (-1, q_l, 1)) )#batch_size * l * 1
        att_v = tf.reshape(att_v, (-1, self.l))

        z = tf.concat(1, [h_p, att_v])

        return z

class Attention_ans:
    def __init__(self, scopeName, l):
        self.l = l
        self.scopeName = scopeName
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scopeName):
            self.V = tf.get_variable("V", initializer = xavier_initializer, shape = (self.l * 2, self.l), dtype = tf.float32)
            self.W_a = tf.get_variable("W_a", initializer = xavier_initializer, shape = (self.l, self.l), dtype = tf.float32)
            self.b_a = tf.get_variable("b_a", initializer = xavier_initializer, shape = (self.l, ), dtype = tf.float32)
            self.v = tf.get_variable("v", initializer = xavier_initializer, shape = (self.l, ), dtype = tf.float32)
            self.c = tf.get_variable("c", initializer = xavier_initializer, shape = (), dtype = tf.float32)
    def attention_one_step(self, H_r_hat, q_l, h_a):
        '''
        paras:
            H_r_hat: batch_size * q_l * (l * 2)
            q_l: passage length + 1
            h_a: batch_size * l
        return:
            beta: batch_size * q_l
            input_lstm_m: batch_size * (2 * l)

        '''

        F_part1 = abc_mul_cd(H_r_hat, self.V, q_l, 2 * self.l, self.l)# batch_size * q_l * l
        F_part2 = tf.matmul(h_a, self.W_a) +  self.b_a # batch_size * l
        F_sum = abc_plus_ac(F_part1, F_part2, q_l, self.l )
        F = tf.nn.tanh( F_sum )# batch_size * q_l * l

        beta = tf.nn.softmax( abc_mul_c(F, self.v, q_l, self.l) + self.c )# batch_size * q_l

        input_lstm_m = tf.matmul( tf.transpose(H_r_hat, (0,2,1)), tf.reshape(beta, (-1, q_l, 1)) )#batch_size * (2*l) * 1
        input_lstm_m = tf.reshape(input_lstm_m, (-1, self.l * 2))

        return beta, input_lstm_m
def pre_layer(lstm_pre, batch_pass, pass_len, batch_ques, ques_len):
    '''
    paras
        lstm_pre:   a LSTM_encoder
                    Note:   input_size = dim_word_v
                            state_size = l
        batch_pass: (None, pass_len, dim_word_v)
        batch_ques: (None, ques_len, dim_word_v)
    return
        H_p: (None, pass_len, l)
        H_q: (None, pass_len, l)


    '''
    H_p = lstm_pre.encode_sequence(batch_pass, pass_len)
    H_q = lstm_pre.encode_sequence(batch_ques, ques_len)
    return H_p, H_q

def match_layer(lstm_m, att_m, H_p, pass_len, H_q, ques_len):
    '''
    paras
        lstm_m:     a LSTM_encoder instance
                    Note: input_size = 2 * l
                          state_size = l
        att_m:      an Attention_match instance
                    Note: l is from here
        H_p:        (None, pass_len, l)
        pass_len:   # of tokens in each passage
        H_q:        (None, pass_len, l)
        ques_len:   # of tokens in each question
    return
        H_r:        (None, pass_len, 2 * l)
    '''
    h_r_lst_r = []
    h_r_lst_l = []

    h_p_lst_r = tf.unstack(H_p, axis = 1)
    h_r_r = tf.zeros(tf.shape(h_p_lst_r[0]))
    mem_r = tf.zeros(tf.shape(h_p_lst_r[0]))

    h_p_lst_l = h_p_lst_r[::-1]
    h_r_l = tf.zeros(tf.shape(h_p_lst_l[0]))
    mem_l = tf.zeros(tf.shape(h_p_lst_l[0]))

    for i in xrange(pass_len):
        z_r = att_m.attention_one_step(H_q, ques_len, h_p_lst_r[i], h_r_r)
        h_r_r_new, mem_r_new = lstm_m.encode_one_step(z_r, h_r_r, mem_r)
        h_r_lst_r.append(h_r_r_new)
        h_r_r = h_r_r_new
        mem_r = mem_r_new

        z_l = att_m.attention_one_step(H_q, ques_len, h_p_lst_l[i], h_r_l)
        h_r_l_new, mem_l_new = lstm_m.encode_one_step(z_l, h_r_l, mem_l)
        h_r_lst_l.append(h_r_l_new)
        h_r_l = h_r_l_new
        mem_l = mem_l_new

    H_r_r = tf.stack(h_r_lst_r)# (pass_len, None, l)
    H_r_l = tf.stack(h_r_lst_l)# (pass_len, None, l)
    H_r = tf.concat(2, [H_r_r, H_r_l]) ## (pass_len, None, 2 * l)
    print H_r_r
    print H_r_l
    print H_r
    H_r = tf.transpose(H_r, (1, 0, 2))
    return H_r
