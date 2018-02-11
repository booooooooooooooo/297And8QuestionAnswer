import tensorflow as tf
import os
from tqdm import tqdm
from datetime import datetime



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
    def __init__(self, input_size, state_size, H_q, ques_sequence_length):
        '''
        input_size = num_units
        state_size = 2 * num_units
        '''
        self._input_size = input_size
        self._state_size = state_size
        self.H_q = H_q
        self.ques_sequence_length = ques_sequence_length

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        '''
        para:
            inputs: (batch_size, _input_size)
            state: (batch_size, _state_size)
        '''
        scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            _input_size = self._input_size
            _state_size = self._state_size
            H_q = self.H_q
            ques_sequence_length = self.ques_sequence_length
            batch_size = tf.shape(H_q)[0]
            ques_max_length = tf.shape(H_q)[1]

            init = tf.contrib.layers.xavier_initializer()
            #attention
            self.W_q = tf.get_variable("W_q", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            self.W_p = tf.get_variable("W_p", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            self.W_r = tf.get_variable("W_r", initializer = init, shape = (_state_size, _input_size), dtype = tf.float32)
            self.b_p = tf.get_variable("b_p", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            self.w = tf.get_variable("w", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            self.b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)

            G = tf.tanh(tf.matmul(H_q, tf.tile(tf.expand_dims(W_q, axis = [0]), [batch_size, 1, 1]))
                    + tf.expand_dims(tf.matmul(inputs, W_p) + tf.matmul(state, W_r) + b_p, [1]))#(batch_size, ques_max_length, _input_size)
            alpha = tf.reshape(tf.matmul(G, tf.tile(tf.reshape(w, [1, _input_size, 1]), [batch_size, 1, 1])), [batch_size, -1])#(batch_size, ques_max_length)
            #TODO:use 10e-1 to mask alpha before appling softmax ??
            alpha = tf.nn.softmax(alpha)#(batch_size, ques_max_length)
            ques_sequence_mask = tf.sequence_mask(ques_sequence_length,ques_max_length, dtype=tf.float32)#(batch_size, ques_max_length)
            alpha = alpha * ques_sequence_mask#(batch_size, ques_max_length)
            att_v = tf.reshape(tf.matmul(tf.expand_dims(alpha, [1]), H_q), [batch_size, _input_size])#(batch_size, _input_size)
            z = tf.concat([h_p, att_v], 1)#(batch_size, 2 * _input_size)

            #gru cell
            #TODO: change 2 * _input_size to _input_size + tf.shape(inputs)[1] ??
            W_z = tf.get_variable('W_z', (2 * _input_size, _state_size), tf.float32, init)
            U_z = tf.get_variable('U_z', (_state_size, _state_size), tf.float32, init)
            b_z = tf.get_variable('b_z', (_state_size,), tf.float32, init)

            W_r = tf.get_variable('W_r', (2 * _input_size, _state_size), tf.float32, init)
            U_r = tf.get_variable('U_r', (_state_size, _state_size), tf.float32, init)
            b_r = tf.get_variable('b_r', (_state_size,), tf.float32, init)


            W_h = tf.get_variable('W_h', (2 * _input_size, _state_size), tf.float32, init)
            U_h = tf.get_variable('U_h', (_state_size, _state_size), tf.float32, init)
            b_h = tf.get_variable('b_h', (_state_size,), tf.float32, init)


            z_gru = tf.nn.sigmoid(tf.matmul(z, W_z) + tf.matmul(state, U_z) + b_z)
            r_gru = tf.nn.sigmoid(tf.matmul(z, W_r) + tf.matmul(state_size, U_r) + b_r)
            new_state_wave = tf.nn.tanh(tf.matmul(z, W_h) + tf.matmul(r_gru * state, U_h) + b_h)
            new_state = z_gru * state + (1 - z_gru) * new_state_wave
            output = new_state

        return output, new_state


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
            self.lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(num_units)
            self.lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(num_units)

        with tf.variable_scope("encode_match"):
            self.lstm_gru_fw = MatchGRUCell(2 * num_units, num_units, H_q, ques_sequence_length)
            self.lstm_gru_bw = MatchGRUCell(2 * num_units, num_units, H_q, ques_sequence_length)

        with tf.variable_scope("decode_ans_ptr"):
            self.V = tf.get_variable("V", shape = (2 * num_units, num_units), dtype = tf.float32, initializer = init)
            self.W_a = tf.get_variable("W_a", shape = (2 * num_units, num_units), dtype = tf.float32, initializer = init)
            self.b_a = tf.get_variable("b_a", shape = (num_units, ), dtype = tf.float32, initializer = init)
            self.v = tf.get_variable("v", shape = (num_units, ), dtype = tf.float32, initializer = init)
            self.c = tf.get_variable("c", shape = (), dtype = tf.float32, initializer = init)

    def encode_preprocess(self, passage, passage_sequence_length, ques, ques_sequence_length, keep_prob,lstm_pre ):
        '''
        return:
            H_p: (batch_size, pass_max_length, 2 *num_units)
            H_q: (batch_size, ques_max_length, 2 * num_units)
        '''

        with tf.variable_scope("preprocessing_layer"):
            H_p, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, passage,
                                       sequence_length = passage_sequence_length,
                                       dtype=tf.float32)
            H_q, _ = tf.nn.dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques,
                                       sequence_length = ques_sequence_length,
                                       dtype=tf.float32)
            H_p = tf.nn.dropout(H_p, keep_prob=keep_prob)
            H_q = tf.nn.dropout(H_q, keep_prob=keep_prob)
        return H_p, H_q

    def encode_match(self, H_p, passage_sequence_length, keep_prob, lstm_gru_fw, lstm_gru_bw):
        '''
        paras
            H_p:        (batch_size, pass_max_length, 2 * num_units)
        return
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        '''
        with tf.variable_scope("encode_match"):
            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                     lstm_gru_bw,
                                                     H_p,
                                                     sequence_length=passage_sequence_length,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)
        return H_r

    def decode_ans_ptr(self, H_r, passage_sequence_length, V, W_a, b_a, v, c, keep_prob):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''


        batch_size = tf.shape(H_r)[0]
        pass_max_length = tf.shape(H_r)[1]
        pass_sequence_mask = tf.sequence_mask(passage_sequence_length,pass_max_length, dtype=tf.float32)#(batch_size, pass_max_length)



        F_s = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                       + b_a)#(batch_size, pass_max_length, num_units)
        beta_s = tf.squeeze(tf.matmul(F_s,
                                      tf.tile(tf.reshape(v, [1, -1, 1]),
                                              [batch_size, 1, 1])))#(batch_size, pass_max_length)
        #TODO:use 10e-1 to mask alpha before appling softmax ??
        beta_s = tf.nn.softmax(beta_s)#(batch_size, pass_max_length)
        beta_s = beta_s * pass_sequence_mask


        h_a = tf.squeeze(tf.matmul(tf.expand_dims(beta_s, [1]), H_r))#(batch_size, 2 * num_units)
        F_e = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                    +  tf.expand_dims(tf.matmul(h_a, W_a) + b_a , [1]))#(batch_size, pass_max_length, num_units)
        beta_e = tf.squeeze(tf.matmul(F_e, tf.tile(tf.reshape(v, [1, -1, 1]), [batch_size, 1, 1])))#(batch_size, pass_max_length)
        #TODO:use 10e-1 to mask alpha before appling softmax ??
        beta_e = tf.nn.softmax(beta_e)#(batch_size, pass_max_length)
        beta_e = beta_e * pass_sequence_mask

        dist = tf.concat([tf.expand_dims(beta_s, [1]), tf.expand_dims(beta_e, [1])], axis = 1)
        return dist


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
