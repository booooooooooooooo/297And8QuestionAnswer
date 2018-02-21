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
regularizer = tf.contrib.layers.l2_regularizer(0.001)
dtype = tf.float32


class matchLSTMcell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, h_question, question_m):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question
        # self.question_m = tf.expand_dims(tf.cast(question_m, tf.int32), axis=[2])
        self.question_m = tf.cast(question_m, tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            # i.e. the batch size
            num_example = tf.shape(self.h_question)[0]

            # TODO: figure out the right way to initialize rnn weights.
            # initializer = tf.contrib.layers.xavier_initializer()
            initializer = tf.uniform_unit_scaling_initializer(1.0)

            W_q = tf.get_variable('W_q', [self.input_size, self.input_size], dtype,
                                  initializer, regularizer=regularizer
                                  )
            W_c = tf.get_variable('W_c', [self.input_size, self.input_size], dtype,
                                  initializer, regularizer=regularizer
                                  )
            W_r = tf.get_variable('W_r', [self._state_size, self.input_size], dtype,
                                  # initializer
                                  initializer, regularizer=regularizer
                                  )
            W_a = tf.get_variable('W_a', [self.input_size, 1], dtype,
                                  initializer, regularizer=regularizer
                                  )
            b_g = tf.get_variable('b_g', [self.input_size], dtype,
                                  tf.zeros_initializer(), regularizer=None)
            b_a = tf.get_variable('b_a', [1], dtype,
                                  tf.zeros_initializer(), regularizer=None)

            wq_e = tf.tile(tf.expand_dims(W_q, axis=[0]), [num_example, 1, 1])
            g = tf.tanh(tf.matmul(self.h_question, wq_e)  # b x q x 2n
                        + tf.expand_dims(tf.matmul(inputs, W_c)
                                         + tf.matmul(state, W_r) + b_g, axis=[1]))
            # TODO:add drop out
            # g = tf.nn.dropout(g, keep_prob=keep_prob)

            wa_e = tf.tile(tf.expand_dims(W_a, axis=0), [num_example, 1, 1])
            # shape: b x q x 1
            a = tf.nn.softmax(tf.squeeze(tf.matmul(g, wa_e) + b_a, axis=[2]))
            # mask out the attention over the padding.
            a = tf.multiply(a, self.question_m)
            question_attend = tf.reduce_sum(tf.multiply(self.h_question, tf.expand_dims(a, axis=[2])), axis=1)

            z = tf.concat([inputs, question_attend], axis=1)

            # NOTE: replace the lstm with GRU.
            # we choose to initialize weight matrix related to hidden to hidden collection with
            # identity initializer.
            W_f = tf.get_variable('W_f', (self._state_size, self._state_size), dtype,
                                  # initializer
                                 initializer, regularizer=regularizer
                                  )
            U_f = tf.get_variable('U_f', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            # initialize b_f with constant 1.0
            b_f = tf.get_variable('b_f', (self._state_size,), dtype,
                                  tf.constant_initializer(1.0),
                                  regularizer=None)
            W_z = tf.get_variable('W_z', (self.state_size, self._state_size), dtype,
                                  # initializer
                                  initializer, regularizer=regularizer
                                  )
            U_z = tf.get_variable('U_z', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            # initialize b_z with constant 1.0
            b_z = tf.get_variable('b_z', (self.state_size,), dtype,
                                  tf.constant_initializer(1.0),
                                  regularizer=None)  # tf.zeros_initializer())
            W_o = tf.get_variable('W_o', (self.state_size, self._state_size), dtype,
                                  # initializer
                                  initializer, regularizer=regularizer
                                  )
            U_o = tf.get_variable('U_o', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            b_o = tf.get_variable('b_o', (self._state_size,), dtype,
                                  tf.constant_initializer(0.0), regularizer=None)

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z)
                                + tf.matmul(state, W_z) + b_z)
            f_t = tf.nn.sigmoid(tf.matmul(z, U_f)
                                + tf.matmul(state, W_f) + b_f)
            o_t = tf.nn.tanh(tf.matmul(z, U_o)
                             + tf.matmul(f_t * state, W_o) + b_o)

            output = z_t * state + (1 - z_t) * o_t
            new_state = output

        return output, new_state


class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, H_q, ques_sequence_mask):
        '''
        input_size = num_units
        state_size = 2 * num_units
        '''
        self._input_size = input_size
        self._state_size = state_size
        self.H_q = H_q
        self.ques_sequence_mask = ques_sequence_mask

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
            alpha = tf.nn.softmax(alpha)#(batch_size, ques_max_length)
            # ques_sequence_mask = tf.sequence_mask(ques_sequence_length,ques_max_length, dtype=tf.float32)#(batch_size, ques_max_length)
            alpha = alpha * ques_sequence_mask#(batch_size, ques_max_length)
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


class MatchLSTMAnsPtr:
    def add_placeholder(self):
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.passage = tf.placeholder(tf.int32, shape = (None, pass_max_length), name = "passage_placeholder")
        self.passage_sequence_length = tf.placeholder(tf.int32, shape = (None,), name = "passage_sequence_length_placeholder")
        self.ques = tf.placeholder(tf.int32, shape = (None, ques_max_length), name = "question_placeholder")
        self.ques_sequence_length = tf.placeholder(tf.int32, shape = (None,), name = "question_sequence_length_placeholder")
        self.ans = tf.placeholder(tf.int32, shape = (None, 2), name = "answer_span_placeholder")
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = "keep_prob_placeholder")

    def get_embed(self):
        passage_embed = tf.nn.embedding_lookup(self.embed_matrix, self.passage)
        ques_embed = tf.nn.embedding_lookup(self.embed_matrix, self.ques)


        return passage_embed, ques_embed

    def encode_preprocess(self, passage_embed, ques_embed):
        '''
        return:
            H_p: (batch_size, pass_max_length, 2 * num_units)
            H_q: (batch_size, ques_max_length, 2 * num_units)
        '''

        with tf.variable_scope("encode_preprocess"):
            lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(self.num_units)

        with tf.variable_scope("encode_preprocess"):
            H_p_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, passage_embed,
                                       sequence_length = self.passage_sequence_length,
                                       dtype=tf.float32)
            H_q_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques_embed,
                                       sequence_length = self.ques_sequence_length,
                                       dtype=tf.float32)
            H_p = tf.concat(H_p_pair, 2)
            H_q = tf.concat(H_q_pair, 2)

        return H_p, H_q

    def encode_match(self, H_p, H_q):
        '''
        paras
            H_p:        (batch_size, pass_max_length, 2 * num_units)
        return
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        '''
        ques_sequence_mask = tf.sequence_mask(self.ques_sequence_length,self.ques_max_length, dtype=tf.float32)

        with tf.variable_scope("encode_match"):
            # lstm_gru_fw = MatchGRUCell(2 * self.num_units, self.num_units, H_q, ques_sequence_mask)
            # lstm_gru_bw = MatchGRUCell(2 * self.num_units, self.num_units, H_q, ques_sequence_mask)
            lstm_gru_fw = matchLSTMcell(2 * self.num_units, self.num_units, H_q, ques_sequence_mask)
            lstm_gru_bw = matchLSTMcell(2 * self.num_units, self.num_units, H_q, ques_sequence_mask)

        with tf.variable_scope("encode_match"):
            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                     lstm_gru_bw,
                                                     H_p,
                                                     sequence_length=self.passage_sequence_length,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)

        return H_r
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
        Wr = tf.get_variable('Wr', [2 * self.num_units, 1 * self.num_units], dtype,
                             initializer, regularizer=regularizer
                             )
        Wh = tf.get_variable('Wh', [2* self.num_units, 1 * self.num_units], dtype,
                             initializer, regularizer=regularizer
                             )
        Wf = tf.get_variable('Wf', [1 * self.num_units, 1], dtype,
                             initializer, regularizer=regularizer
                             )
        br = tf.get_variable('br', [1 * self.num_units], dtype,
                             tf.zeros_initializer())
        bf = tf.get_variable('bf', [1, ], dtype,
                             tf.zeros_initializer())

        wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [shape_Hr[0], 1, 1])
        f = tf.tanh(tf.matmul(H_r, wr_e) + br)

        # TODO: add dropout
        f = tf.nn.dropout(f, keep_prob=keep_prob)

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

        # return s_score, e_score

        dist = tf.concat([tf.expand_dims(s_prob, [1]), tf.expand_dims(e_prob, [1])], axis = 1)
        print s_score
        print e_score
        print dist
        return dist

    def decode_ans_ptr(self, H_r):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("decode_ans_ptr"):
            V = tf.get_variable("V", shape = (2 * self.num_units, self.num_units), dtype = tf.float32, initializer = init)
            W_a = tf.get_variable("W_a", shape = (2 * self.num_units, self.num_units), dtype = tf.float32, initializer = init)
            b_a = tf.get_variable("b_a", shape = (self.num_units, ), dtype = tf.float32, initializer = init)
            v = tf.get_variable("v", shape = (self.num_units, ), dtype = tf.float32, initializer = init)
            c = tf.get_variable("c", shape = (), dtype = tf.float32, initializer = init)


        batch_size = tf.shape(H_r)[0]
        pass_max_length = self.pass_max_length
        pass_sequence_mask = tf.sequence_mask(self.passage_sequence_length,pass_max_length, dtype=tf.float32)#(batch_size, pass_max_length)



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
        passage_embed, ques_embed = self.get_embed()
        H_p, H_q = self.encode_preprocess(passage_embed,  ques_embed)
        H_r = self.encode_match(H_p, H_q)
        # dist = self.decode_ans_ptr(H_r)
        context_m = tf.sequence_mask(self.passage_sequence_length, self.pass_max_length, dtype=tf.float32)
        dist = self.decode(H_r, context_m, self.keep_prob)

        self.dist = tf.identity(dist, name = "dist")#dist is used in validation and test

    def add_loss_function(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.dist, labels=self.ans))

    def add_train_op(self, clip_norm, optimizer, lr):

        if optimizer == "adam":
            optimizer_func = tf.train.AdamOptimizer(lr)
        elif optimizer == "sgd":
            optimizer_func = tf.train.GradientDescentOptimizer(lr)
        else:
            raise ValueError('Parameters are wrong')

        gradients, variables = zip(*optimizer_func.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()

        self.train_op = train_op

    def __init__(self, embed_matrix, pass_max_length, ques_max_length, embed_size, num_units, clip_norm, optimizer, lr, n_epoch):
        #Train, valid and test data must be consistent on these parameters.
        self.embed_matrix = embed_matrix
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length#not sure
        self.embed_size = embed_size
        #parameter used by the graph. It is not related to data.
        self.num_units = num_units
        self.clip_norm = clip_norm
        self.optimizer = optimizer
        self.lr = lr
        self.n_epoch = n_epoch
        #build the graph
        self.add_placeholder()
        self.add_predicted_dist()
        self.add_loss_function()
        self.add_train_op(clip_norm, optimizer, lr)

    def run_batch(self, sess, batch, dir_output):
        passage, passage_sequence_length, ques ,ques_sequence_length, ans, keep_prob = batch
        _ , batch_loss = sess.run([self.train_op, self.loss], {self.passage : passage,
                                                          self.passage_sequence_length : passage_sequence_length,
                                                          self.ques : ques,
                                                          self.ques_sequence_length : ques_sequence_length,
                                                          self.ans : ans,
                                                          self.keep_prob : keep_prob})



        graph_file = os.path.join(dir_output, "graphes/", datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S"))
        tf.train.Saver().save(sess, graph_file )


        batch_stat = {"batch_train_loss": batch_loss, "graph_file": graph_file}


        return batch_stat
    def validate_on_hot(self, sess, small_validation_dataset):
        #TODO
        passage, passage_sequence_length, ques, ques_sequence_length, ans, passage_text, ques_text, ans_texts, voc = small_validation_dataset
        dist, loss = sess.run([self.dist, self.loss], {self.passage : passage,
                                                      self.passage_sequence_length : passage_sequence_length,
                                                      self.ques : ques,
                                                      self.ques_sequence_length : ques_sequence_length,
                                                      self.ans : ans,
                                                      self.keep_prob : 1.0})
        predict_ans = np.argmax(dist, axis = 2)#(batch_size, 2)
        predict_ans_texts = []
        f1 = 0.0
        em = 0.0
        for i in xrange(predict_ans.shape[0]):
            predict_ans_tokens = []
            for j in range(predict_ans[i][0], predict_ans[i][1]):
                predict_ans_tokens.append(voc[j])
            predict_ans_text = " ".join(predict_ans_tokens)
            f1 += f1_score(predict_ans_text, ans_texts[i])
            em += exact_match_score(predict_ans_text, ans_texts[i])
            predict_ans_texts.append(predict_ans_text)
        f1 = f1 / predict_ans.shape[0]
        em = em / predict_ans.shape[0]

        qas = []
        for i in xrange(predict_ans.shape[0]):
            qas.append((passage_text[i], ques_text[i], ans_texts[i], predict_ans_texts[i]))
        validation_result = {"loss": loss, "f1":f1, "em":em, "qas":qas}
        return validation_result

    def run_epoch(self, sess, batches, small_validation_dataset, dir_output):
        epoch_stat = []
        for i in tqdm(xrange(len(batches)), desc = "Under training") :
            batch_stat = self.run_batch(sess, batches[i], dir_output)
            epoch_stat.append(batch_stat)

            print "{}/{}th batch_train_loss: {}".format(i, len(batches), batch_stat["batch_train_loss"])
            if i % 20 == 0:
                validation_result = self.validate_on_hot(sess, small_validation_dataset)
                print "{}/{}th valid loss: {}".format(i, len(batches), validation_result["loss"])
                print "f1 : {}".format(validation_result["f1"])
                print "em : {}".format(validation_result["em"])
                print validation_result["qas"][0][0]
                print validation_result["qas"][0][1]
                print validation_result["qas"][0][2]
                print validation_result["qas"][0][3]
        #TODO: print reader-friendly epoch result
        return epoch_stat
    def fit(self, sess, batches, small_validation_dataset, dir_output):
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)
        if not os.path.isdir(os.path.join(dir_output, "graphes/")):
            os.makedirs(os.path.join(dir_output, "graphes/"))

        #Just in case. sess para should be already initialized
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"

        stats = []
        for epoch in xrange(self.n_epoch) :
            print "Epoch {}".format(epoch)
            epoch_stat = self.run_epoch(sess, batches, small_validation_dataset, dir_output)
            stats.append(epoch_stat)
        #TODO: print reader-friendly final result
        return stats
