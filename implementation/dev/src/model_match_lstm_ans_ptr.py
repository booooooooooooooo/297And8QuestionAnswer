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
            ques_sequence_mask = tf.sequence_mask(ques_sequence_length,ques_max_length, dtype=tf.float32)#(batch_size, ques_max_length)
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

        passage_embed = tf.nn.dropout(passage_embed, keep_prob=self.keep_prob)
        ques_embed = tf.nn.dropout(ques_embed, keep_prob=self.keep_prob)

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

            H_p = tf.nn.dropout(H_p, keep_prob=self.keep_prob)
            H_q = tf.nn.dropout(H_q, keep_prob=self.keep_prob)
        return H_p, H_q

    def encode_match(self, H_p, H_q):
        '''
        paras
            H_p:        (batch_size, pass_max_length, 2 * num_units)
        return
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        '''

        with tf.variable_scope("encode_match"):
            lstm_gru_fw = MatchGRUCell(2 * self.num_units, self.num_units, H_q, self.ques_sequence_length)
            lstm_gru_bw = MatchGRUCell(2 * self.num_units, self.num_units, H_q, self.ques_sequence_length)

        with tf.variable_scope("encode_match"):
            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                     lstm_gru_bw,
                                                     H_p,
                                                     sequence_length=self.passage_sequence_length,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)
            H_r = tf.nn.dropout(H_r, keep_prob=self.keep_prob)
        return H_r

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

        #TODO: add dropout?

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
        dist = self.decode_ans_ptr(H_r)

        self.dist = tf.identity(dist, name = "dist")#dist is used in validation and test

    def add_loss_function(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.dist, labels=self.ans))

    def add_train_op(self, clip_norm, optimizer, lr):
        decay_lr = tf.train.exponential_decay(lr, tf.Variable(0, trainable=False), 1000, 0.96, staircase=True)

        if optimizer == "adam":
            optimizer_func = tf.train.AdamOptimizer(decay_lr)
        elif optimizer == "sgd":
            optimizer_func = tf.train.GradientDescentOptimizer(decay_lr)
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



        graph_file = os.path.join(dir_output, "graphes/", str(datetime.now()))
        tf.train.Saver().save(sess, graph_file )

        #TODO: do validation per batch?

        batch_stat = {"batch_train_loss": batch_loss, "graph_file": graph_file}

        #TODO:print reader-friendly batch result
        print "batch_train_loss: {}".format(batch_loss)

        return batch_stat
    def run_epoch(self, sess, batches, dir_output):
        epoch_stat = []
        for i in tqdm(xrange(len(batches)), desc = "Under training") :
            batch_stat = self.run_batch(sess, batches[i], dir_output)
            epoch_stat.append(batch_stat)
        #TODO: print reader-friendly epoch result
        return epoch_stat
    def fit(self, sess, batches, dir_output):
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
            epoch_stat = self.run_epoch(sess, batches, dir_output)
            stats.append(epoch_stat)
        #TODO: print reader-friendly final result
        return stats
