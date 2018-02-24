import tensorflow as tf
import os
from tqdm import tqdm
import datetime
import random
import numpy as np
import json

from evaluate_v_1_1 import f1_score, exact_match_score
from util_data import *


class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, H_q, ques_mask):
        '''
        input_size: size of encoded passage vector
        state_size = size of this cell's hidden units
        '''
        self._input_size = input_size
        self._state_size = state_size
        self.H_q = H_q
        self.ques_mask = ques_mask

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
            ques_mask = self.ques_mask
            batch_size = tf.shape(H_q)[0]
            ques_max_length = tf.shape(H_q)[1]

            init = tf.contrib.layers.xavier_initializer()
            #attention
            W_q = tf.get_variable("W_q", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_p = tf.get_variable("W_p", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_r_0 = tf.get_variable("W_r_0", initializer = init, shape = (_state_size, _input_size), dtype = tf.float32)
            b_p = tf.get_variable("b_p", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            w = tf.get_variable("w", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)

            G = tf.tanh(tf.matmul(H_q, tf.tile(tf.expand_dims(W_q, axis = [0]), [batch_size, 1, 1]))
                    + tf.expand_dims(tf.matmul(inputs, W_p) + tf.matmul(state, W_r_0) + b_p, [1]))#(batch_size, ques_max_length, _input_size)
            alpha = tf.reshape(tf.matmul(G, tf.tile(tf.reshape(w, [1, _input_size, 1]), [batch_size, 1, 1])), [batch_size, -1])#(batch_size, ques_max_length)
            #TODO:use 10e-1 to mask alpha before appling softmax ??
            alpha = tf.nn.softmax(alpha) * ques_mask #(batch_size, ques_max_length)
            att_v = tf.reshape(tf.matmul(tf.expand_dims(alpha, [1]), H_q), [batch_size, _input_size])#(batch_size, _input_size)
            z = tf.concat([inputs, att_v], 1)#(batch_size, 2 * _input_size)

            #gru cell
            #TODO: change 2 * _input_size to _input_size + tf.shape(inputs)[1] ??
            W_z = tf.get_variable('W_z', (2 * _input_size, _state_size), tf.float32, init)
            U_z = tf.get_variable('U_z', (_state_size, _state_size), tf.float32, init)
            b_z = tf.get_variable('b_z', (_state_size,), tf.float32, init)

            W_r_1 = tf.get_variable('W_r_1', (2 * _input_size, _state_size), tf.float32, init)
            U_r = tf.get_variable('U_r', (_state_size, _state_size), tf.float32, init)
            b_r = tf.get_variable('b_r', (_state_size,), tf.float32, init)


            W_h = tf.get_variable('W_h', (2 * _input_size, _state_size), tf.float32, init)
            U_h = tf.get_variable('U_h', (_state_size, _state_size), tf.float32, init)
            b_h = tf.get_variable('b_h', (_state_size,), tf.float32, init)


            z_gru = tf.nn.sigmoid(tf.matmul(z, W_z) + tf.matmul(state, U_z) + b_z)
            r_gru = tf.nn.sigmoid(tf.matmul(z, W_r_1) + tf.matmul(state, U_r) + b_r)
            new_state_wave = tf.nn.tanh(tf.matmul(z, W_h) + tf.matmul(r_gru * state, U_h) + b_h)
            new_state = z_gru * state + (1 - z_gru) * new_state_wave
            output = new_state

        return output, new_state


class Model:
    def __init__(self, embed_matrix, pass_max_length, ques_max_length, embed_size, num_units, clip_norm, lr, n_epoch, reg_scale):
        #Train, valid and test data must be consistent on these parameters.
        self.embed_matrix = embed_matrix
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length
        self.embed_size = embed_size
        #not related to data.
        self.num_units = num_units
        self.clip_norm = clip_norm
        self.lr = lr
        self.regularizer = tf.contrib.layers.l2_regularizer(reg_scale)
        self.n_epoch = n_epoch
        #build the graph
        self.add_placeholder()
        self.add_predicted_dist()
        self.add_loss_function()
        self.add_train_op()

    def add_placeholder(self):
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.passage = tf.placeholder(tf.int32, shape = (None, pass_max_length), name = "passage_placeholder")
        self.passage_mask = tf.placeholder(tf.float32, shape = (None, pass_max_length), name = "passage_sequence_length_placeholder")
        self.ques = tf.placeholder(tf.int32, shape = (None, ques_max_length), name = "question_placeholder")
        self.ques_mask = tf.placeholder(tf.float32, shape = (None, ques_max_length), name = "question_sequence_length_placeholder")
        self.answer_s = tf.placeholder(tf.int32, (None,), name = "answer_start")
        self.answer_e = tf.placeholder(tf.int32, (None,), name = "answer_end")


    def encode(self, embed_matrix, passage, passage_mask, ques, ques_mask, num_units, regularizer):
        '''
        return:
            H_p: (batch_size, pass_max_length, 2 * num_units)
            H_q: (batch_size, ques_max_length, 2 * num_units)
        '''
        with tf.variable_scope("embedding"):
            passage_embed = tf.nn.embedding_lookup(embed_matrix, passage)
            ques_embed = tf.nn.embedding_lookup(embed_matrix, ques)

        with tf.variable_scope("encode_preprocess"):
            lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(num_units)
            lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(num_units)

            H_p_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, passage_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(passage_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_q_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(ques_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_p = tf.concat(H_p_pair, 2)
            H_q = tf.concat(H_q_pair, 2)


        with tf.variable_scope("encode_match"):
            lstm_gru_fw = MatchGRUCell(2 * num_units, 2 * num_units, H_q, ques_mask)
            lstm_gru_bw = MatchGRUCell(2 * num_units, 2 * num_units, H_q, ques_mask)


            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                         lstm_gru_bw,
                                                         H_p,
                                                         sequence_length=tf.reduce_sum(tf.cast(passage_mask, tf.int32), axis=1),
                                                         dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)

        return H_r


    def decode(self, H_r, pass_max_length, passage_mask, num_units, regularizer):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''
        batch_size = tf.shape(H_r)[0]

        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decode_ans_ptr"):
            V = tf.get_variable("V", shape = (4 * num_units, 2 * num_units), dtype = tf.float32, initializer = init)
            W_a = tf.get_variable("W_a", shape = (4 * num_units, 2 * num_units), dtype = tf.float32, initializer = init)
            b_a = tf.get_variable("b_a", shape = (2 * num_units, ), dtype = tf.float32, initializer = init)
            v = tf.get_variable("v", shape = (2 * num_units, ), dtype = tf.float32, initializer = init)
            c = tf.get_variable("c", shape = (), dtype = tf.float32, initializer = init)


            F_s = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                           + b_a)#(batch_size, pass_max_length, num_units)
            beta_s = tf.squeeze(tf.matmul(F_s,
                                          tf.tile(tf.reshape(v, [1, -1, 1]),
                                                  [batch_size, 1, 1])))#(batch_size, pass_max_length)





            prob_s = tf.nn.softmax(beta_s) * passage_mask#(batch_size, pass_max_length)
            h_a = tf.squeeze(tf.matmul(tf.expand_dims(prob_s, [1]), H_r))#(batch_size, 2 * num_units)
            F_e = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                        +  tf.expand_dims(tf.matmul(h_a, W_a) + b_a , [1]))#(batch_size, pass_max_length, num_units)
            beta_e = tf.squeeze(tf.matmul(F_e, tf.tile(tf.reshape(v, [1, -1, 1]), [batch_size, 1, 1])))#(batch_size, pass_max_length)




        return beta_s, beta_e



    def add_predicted_dist(self):
        H_r = self.encode(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units, self.regularizer)
        self.beta_s, self.beta_e = self.decode(H_r, self.pass_max_length, self.passage_mask, self.num_units, self.regularizer)

    def add_loss_function(self):
        loss_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.beta_s, labels=self.answer_s))
        loss_e = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.beta_e, labels=self.answer_e)
                                    )
        reg_losses = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_losses)

        self.loss = (loss_s + loss_e) / 2.0 + reg_term


    def add_train_op(self):
        optimizer_func = tf.train.AdamOptimizer(self.lr)

        gradients, variables = zip(*optimizer_func.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()

        self.train_op = train_op

    '''Functions below are used for train and validation'''
    def predict(self, sess, passage, passage_mask, ques ,ques_mask):
        dist_s, dist_e = sess.run([self.beta_s, self.beta_e], {self.passage: passage,
                                                               self.passage_mask: passage_mask,
                                                               self.ques: ques,
                                                               self.ques_mask: ques_mask})
        idx_s = np.argmax(dist_s, axis=1)
        idx_e = np.argmax(dist_e, axis=1)

        return idx_s, idx_e
    def validate(self, sess, data_tuple, sample_size):
        passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = data_tuple
        bound = min(sample_size, len(passage))
        passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text = passage[0: bound],\
                                                                                  passage_mask[0: bound],\
                                                                                  ques[0: bound],\
                                                                                  ques_mask[0: bound],\
                                                                                  answer_s[0: bound],\
                                                                                  answer_e[0: bound],\
                                                                                  answer_text[0: bound]

        loss = sess.run(self.loss, {self.passage: passage,
                                    self.passage_mask: passage_mask,
                                    self.ques: ques,
                                    self.ques_mask: ques_mask,
                                    self.answer_s: answer_s,
                                    self.answer_e: answer_e})
        idx_s, idx_e = self.predict(sess, passage, passage_mask, ques ,ques_mask)
        predict_text = predict_ans_text(idx_s, idx_e, passage, voc)
        f1 = 0.0
        em = 0.0
        for i in xrange(len(answer_text)):
            f1 += f1_score(predict_text[i].decode('utf8'), answer_text[i].decode('utf8'))
            em += exact_match_score(predict_text[i].decode('utf8'), answer_text[i].decode('utf8'))
        f1 /= len(answer_text)
        em /= len(answer_text)

        return loss, f1, em

    def fit(self, sess, train_data, valid_data, batch_size, sample_size, dir_output):
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)
        if not os.path.isdir(os.path.join(dir_output, "graphes/")):
            os.makedirs(os.path.join(dir_output, "graphes/"))

        #Just in case. sess para should be already initialized
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"

        for epoch in xrange(self.n_epoch):
            train_stat = []
            passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = train_data
            batches = get_batches(passage, passage_mask, ques, ques_mask, answer_s, answer_e, batch_size)
            for num in xrange(len(batches)):
                b_passage, b_passage_mask, b_ques, b_ques_mask, b_answer_s, b_answer_e = batches[num]
                _ , batch_loss = sess.run([self.train_op, self.loss], {self.passage: b_passage,
                                                                       self.passage_mask: b_passage_mask,
                                                                       self.ques: b_ques,
                                                                       self.ques_mask: b_ques_mask,
                                                                       self.answer_s: b_answer_s,
                                                                       self.answer_e: b_answer_e})


                print "epoch: {}, batch: {} / {}, batch_loss: {}".format(epoch, num, len(batches), batch_loss)
                if num % 25 == 0:
                    graph_file = os.path.join(dir_output, "graphes/", datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S"))
                    tf.train.Saver().save(sess, graph_file)
                    train_loss, train_f1, train_em = self.validate(sess, train_data, sample_size)
                    valid_loss, valid_f1, valid_em = self.validate(sess, valid_data, sample_size)
                    batch_stat = {"epoch": str(epoch), "batch": str(num), "batch_loss" : str(batch_loss), \
                                  "sample_size": str(sample_size), "train_loss": str(train_loss),\
                                  "train_f1" : str(train_f1), "train_em": str(train_em), \
                                  "valid_loss": str(valid_loss), "valid_f1": str(valid_f1), \
                                  "valid_em": str(valid_em) , "graph_file" : graph_file}
                    train_stat.append(batch_stat)
                    print "================"
                    print "validation_sample_size: {}".format(sample_size)
                    print "Sample train_loss: {}, train_f1 : {}, train_em : {}".format( train_loss, train_f1, train_em)
                    print "Sample valid_loss: {}, valid_f1: {}, valid_em: {}".format(valid_loss, valid_f1, valid_em)
                    print "================"
                

            with open(os.path.join(dir_output, "epoch-{}-train-stat-{}".format(epoch, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S"))), 'w') as f:
                f.write(json.dumps(train_stat))

        print "Finish trainning! Congs!"
