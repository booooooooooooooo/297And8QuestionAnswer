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
        # Assume all sequences have same length. Padding work is done in data related files.
        states = []
        batch_states = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        batch_memories = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        for i in xrange(seq_len):
            new_batch_states, new_batch_memories = self.encode_one_step(batch_inputs_sequence[:,i], batch_states, batch_memories)
            states.append(new_batch_states)
            batch_states, batch_memories = new_batch_states, new_batch_memories
        return tf.transpose( tf.stack(states) , (1, 0, 2))

class Attention_match:
    def __init__(self, scopeName, state_size):
        self.state_size = state_size
        self.scopeName = scopeName
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scopeName):
            self.W_q = tf.get_variable("W_q", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.W_p = tf.get_variable("W_p", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.W_r = tf.get_variable("W_r", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_p = tf.get_variable("b_p", initializer = xavier_initializer, shape = (self.state_size, ), dtype = tf.float32)
            self.w = tf.get_variable("w", initializer = xavier_initializer, shape = (self.state_size, ), dtype = tf.float32)
            self.b = tf.get_variable("b", initializer = xavier_initializer, shape = (), dtype = tf.float32)
    def attention_one_step(self, H_q, q_l, h_p, h_r):
        '''
        paras:
            H_q: batch_size * q_l * state_size
            h_p: batch_size * state_size
            h_r: batch_size * state_size
        return:
            z: batch_size * (2 * state_size)
        '''

        G_part1 = abc_mul_cd(H_q, self.W_q, q_l, self.state_size, self.state_size)# batch_size * q_l * state_size
        G_part2 = tf.matmul(h_p, self.W_p) + tf.matmul(h_r, self.W_r) + self.b_p # batch_size * state_size
        G_sum = abc_plus_ac(G_part1, G_part2, q_l, self.state_size)
        G = tf.nn.tanh( G_sum )# batch_size * q_l * state_size

        alpha = tf.nn.softmax( abc_mul_c(G, self.w, q_l, self.state_size) + self.b )# batch_size * q_l

        att_v = tf.matmul( tf.transpose(H_q, (0,2,1)), tf.reshape(alpha, (-1, q_l, 1)) )#batch_size * state_size * 1
        att_v = tf.reshape(att_v, (-1, self.state_size))

        z = tf.concat(1, [h_p, att_v])

        return z

class Attention_ans:
    def __init__(self, scopeName, state_size):
        self.state_size = state_size
        self.scopeName = scopeName
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scopeName):
            self.V = tf.get_variable("V", initializer = xavier_initializer, shape = (self.state_size * 2, self.state_size), dtype = tf.float32)
            self.W_a = tf.get_variable("W_a", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_a = tf.get_variable("b_a", initializer = xavier_initializer, shape = (self.state_size, ), dtype = tf.float32)
            self.v = tf.get_variable("v", initializer = xavier_initializer, shape = (self.state_size, ), dtype = tf.float32)
            self.c = tf.get_variable("c", initializer = xavier_initializer, shape = (), dtype = tf.float32)
    def attention_one_step(self, H_r_hat, q_l, h_a):
        '''
        paras:
            H_r_hat: batch_size * q_l * (state_size * 2)
            q_l: passage length + 1
            h_a: batch_size * state_size
        return:
            beta: batch_size * q_l
            state: batch_size * (2 * state_size)

        '''

        F_part1 = abc_mul_cd(H_r_hat, self.V, q_l, 2 * self.state_size, self.state_size)# batch_size * q_l * state_size
        F_part2 = tf.matmul(h_a, self.W_a) +  self.b_a # batch_size * state_size
        F_sum = abc_plus_ac(F_part1, F_part2, q_l, self.state_size )
        F = tf.nn.tanh( F_sum )# batch_size * q_l * state_size

        beta = tf.nn.softmax( abc_mul_c(F, self.v, q_l, self.state_size) + self.c )# batch_size * q_l

        state_lstm = tf.matmul( tf.transpose(H_r_hat, (0,2,1)), tf.reshape(beta, (-1, q_l, 1)) )#batch_size * (2*state_size) * 1
        state_lstm = tf.reshape(state_lstm, (-1, self.state_size * 2))

        return beta, state_lstm
