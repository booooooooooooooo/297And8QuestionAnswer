'''
Used by tune.py to train better tensor graph.
'''
import tensorflow as tf

from util import *

'''
TODO:
embed_matrix placeholder
mask placeholder is boolean type. Only use one mask, and use util to get needed ones.
regularization
gradient decent clipping


efficient training: nce etc.


mark public and private def
give nice names to important tensors, include embed_size, batch_size, num_units, lr, n_epoch etc. in title of saved model
'''
class Model:
    def __init__(self, batch_size, pass_max_length, ques_max_length, embed_size, num_units):
        #parameters used by the graph. Train, valid and test data must be consistent on these parameters.
        self.batch_size = batch_size
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length#not sure
        self.embed_size = embed_size
        #A parameter used by the graph. It is not related to data.
        self.num_units = num_units
        #build the graph
        self.build()
    def __add_placeholder(self):
        batch_size = self.batch_size
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.ques = tf.placeholder(tf.int32, shape = (batch_size, ques_max_length, embed_size), name = "question_placeholder")
        self.ques_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "question_sequence_length_placeholder")
        self.passage = tf.placeholder(tf.int32, shape = (batch_size, pass_max_length, embed_size), name = "passage_placeholder")
        self.passage_sequence_length = tf.placeholder(tf.int32, shape = (batch_size), name = "passage_sequence_length_placeholder")
        self.ans = tf.placeholder(tf.int32, shape = (batch_size, 2), name = "answer_span_placeholder")
    def __add_variables(self):
        embed_size = self.embed_size
        num_units = self.num_units

        init = tf.contrib.layers.xavier_initializer()

        #Preprocessing Layer
        with tf.variable_scope("preprocessing_layer"):
            self.lstm_pre = tf.contrib.rnn.BasicLSTMCell(num_units)
        #Match-LSTM Layer
        with tf.variable_scope("match_ayer"):
            with tf.variable_scope("Attention"):
                self.W_q = tf.get_variable("W_q", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_p = tf.get_variable("W_p", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.W_r = tf.get_variable("W_r", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_p = tf.get_variable("b_p", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.w = tf.get_variable("w", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.lstm_match = tf.contrib.rnn.BasicLSTMCell(num_units)
        #Answer Pointer Layer
        with tf.variable_scope("answer_pointer_layer"):
            with tf.variable_scope("Attention"):
                self.V = tf.get_variable("V", initializer = init, shape = (num_units * 2, num_units), dtype = tf.float32)
                self.W_a = tf.get_variable("W_a", initializer = init, shape = (num_units, num_units), dtype = tf.float32)
                self.b_a = tf.get_variable("b_a", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.v = tf.get_variable("v", initializer = init, shape = (num_units, ), dtype = tf.float32)
                self.c = tf.get_variable("c", initializer = init, shape = (), dtype = tf.float32)
            with tf.variable_scope("LSTM"):
                self.lstm_ans = tf.contrib.rnn.BasicLSTMCell(num_units)
    def __pre_layer(self):
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

        H_p, _ = tf.nn.dynamic_rnn(lstm_pre, passage, sequence_length = passage_sequence_length,
                                           initial_state=initial_state,
                                           dtype=tf.float32)
        H_q, _ = tf.nn.dynamic_rnn(lstm_pre, ques, sequence_length = ques_sequence_length,
                                           initial_state=initial_state,
                                           dtype=tf.float32)
        return H_p, H_q

    def __match_attention(self, H_q, h_p, h_r):
        '''
        paras:
            H_q: (batch_size, ques_max_length, num_units)
            h_p: (batch_size, num_units)
            h_r: (batch_size, num_units)
        return:
            z: (batch_size, 2 * num_units)
        '''
        batch_size = self.batch_size
        num_units = self.num_units

        ques_mask_matrix = self.ques_mask_matrix

        W_q = self.W_q
        W_p = self.W_p
        W_r = self.W_r
        b_p = self.b_p
        w = self.w
        b = self.b

        G_1 = tf.matmul( tf.reshape(H_q, (-1, num_units)), W_q)#(batch_size * ques_max_length, num_units)
        G_1 = tf.reshape(G_1, (batch_size, -1, num_units))#(batch_size, ques_max_length, num_units)
        G_1 = tf.transpose(G_1, (0, 2, 1))#(batch_size, num_units, ques_max_length)
        G_1 = tf.reshape(G_1, (batch_size * num_units, -1))#(batch_size * num_units, ques_max_length)
        G_2 = tf.matmul(h_p, W_p) + tf.matmul(h_r, W_r) + b_p #(batch_size, num_units)
        G_2 = tf.reshape(G_2, (-1, 1))#(batch_size * num_units, 1)
        G = tf.nn.tanh( G_1 + G_2 )#(batch_size * num_units, ques_max_length)
        G = tf.reshape(G, (batch_size, num_units, -1))#(batch_size, num_units, ques_max_length)
        G = tf.transpose(G, (0, 2, 1))#(batch_size, ques_max_length, num_units)
        G = tf.reshape(G, (-1, num_units))#(batch_size * ques_max_length, num_units)

        alpha = tf.matmul(G, tf.reshape(w, (-1, 1))) + b #(batch_size * ques_max_length, 1)
        alpha = tf.reshape(alpha, (batch_size, -1))#(batch_size, ques_max_length)
        #masking
        alpha = alpha * ques_mask_matrix#(batch_size, ques_max_length)
        #softmax
        alpha = tf.nn.softmax(alpha)#(batch_size, ques_max_length)
        alpha = tf.reshape(alpha, (batch_size, -1, 1))#(batch_size, ques_max_length, 1)

        att_state = tf.matmul(tf.transpose(H_q, (0, 2, 1)), alpha)#(batch_size, num_units, 1)
        att_state = tf.reshape(att_state, (batch_size, num_units))#(batch_size, num_units)


        z = tf.concat([h_p, att_state], 1)#(batch_size, 2 * num_units)

        return z
    def __match_one_direct(self, H_p, H_q, direction):
        '''
        paras
            H_p:        (None, pass_max_length, num_units)
            H_q:        (None, ques_max_length, num_units)
        return
            H_r_one_direct:        (None, pass_max_length, num_units)
        '''

        num_units = self.num_units
        batch_size = self.batch_size
        pass_max_length = self.pass_max_length


        h_p_lst = tf.unstack(H_p, axis = 1)

        h_r_lst = []
        with tf.variable_scope("match_ayer"):
            with tf.variable_scope("LSTM"):
                lstm_match = self.lstm_match
                stateTuple = lstm_match.zero_state(batch_size, tf.float32)
                if direction == "right":
                    for i in xrange(pass_max_length):
                        h_p = h_p_lst[i]
                        z_i = self.match_attention(H_q, h_p, stateTuple[1])
                        _, stateTuple = lstm_match(z_i, stateTuple)
                        h_r_lst.append(stateTuple[1])
                elif direction == "left":
                    #TODO
                else:
                    raise ValueError('direction must be right or left')
        H_r_one_direct = tf.stack(h_r_lst)# (pass_max_length, None, num_units)
        H_r_one_direct = tf.transpose(H_r_one_direct, (1, 0, 2))# (None, pass_max_length, num_units)
        return H_r_one_direct
    def __match_layer(self, H_p, H_p_rev, H_q):
        '''
        paras
            H_r_right:        (None, pass_max_length, num_units)
            H_r_left:         (None, ques_max_length, num_units)
        return
            H_r:              (None, pass_max_length, 2 * num_units)
        '''
        passage_mask_cube = self.passage_mask_cube#(None, pass_max_length, num_units)
        num_units = self.num_units

        H_r_right = self.match_one_direct(H_p, H_q)#(None, pass_max_length, num_units)
        H_r_left = self.match_one_direct(H_p_rev, H_q)#(None, pass_max_length, num_units)
        #masking
        H_r_right *= passage_mask_cube#(None, pass_max_length, num_units)
        H_r_left *= passage_mask_cube#(None, pass_max_length, num_units)
        #concat
        H_r = tf.concat([H_r_right, H_r_left], 2)#(None, pass_max_length, 2 * num_units)
        return H_r



    def __answer_attention(self, H_r, h_a):
        '''
        paras:
            H_r: (batch_size, pass_max_length, 2 * num_units)
            h_a: (batch_size, num_units)
        return:
            beta: (batch_size, pass_max_length)
            input_lstm: (batch_size, 2 * num_units)

        '''
        pass_max_length = self.pass_max_length
        num_units = self.num_units
        passage_mask_matrix = self.passage_mask_matrix#(None, pass_max_length)

        V = self.V#(num_units * 2, num_units)
        W_a = self.W_a#(num_units, num_units)
        b_a = self.b_a#(num_units, )
        v = self.v#(num_units, )
        c = self.c#()

        F_1 = abc_mul_cd(H_r, V, pass_max_length, 2 * num_units, num_units)# (batch_size, pass_max_length, num_units)
        F_2 = tf.matmul(h_a, W_a) + b_a # (batch_size, num_units)
        F_sum = abc_plus_ac(F_1, F_2, pass_max_length, num_units )
        F = tf.nn.tanh( F_sum )# (batch_size, pass_max_length, num_units)

        beta = abc_mul_c(F, v, pass_max_length, num_units) + c#(batch_size, pass_max_length)
        #masking
        beta *= passage_mask_matrix#(batch_size, pass_max_length)
        #softmax
        beta = tf.nn.softmax(beta)#(batch_size, pass_max_length)

        input_lstm = tf.matmul( tf.transpose(H_r, (0,2,1)), tf.reshape(beta, (-1, pass_max_length, 1)) )#(batch_size, 2 * num_units, 1)
        input_lstm = tf.reshape(input_lstm, (-1, 2 * num_units))

        return beta, input_lstm
    def __answer_layer(self, H_r):
        '''
        paras
            H_r:        (None, pass_max_length, 2 * num_units)
        return
            dist:       (None, 2, pass_max_length)
        '''
        num_units = self.num_units
        batch_size = self.batch_size

        dist_lst = []
        with tf.variable_scope("answer_pointer_layer"):
            with tf.variable_scope("LSTM"):
                lstm_ans = self.lstm_ans
                stateTuple = lstm_ans.zero_state(batch_size, tf.float32)
                for i in xrange(2):
                    beta, input_lstm = self.answer_attention(H_r, stateTuple[1])
                    _, stateTuple = lstm_ans(input_lstm, stateTuple)
                    dist_lst.append(beta)
        dist = tf.stack(dist_lst)#(2, None, pass_max_length)
        dist = tf.transpose(dist, (1,0,2))#(None, 2, pass_max_length)
        return dist

    def __add_predicted_dist(self):
        #embdding
        pass_embed, pass_embed_rev, ques_embed = self.embed_layer()
        #Preprocessing Layer
        H_p, H_p_rev, H_q = self.pre_layer(pass_embed, pass_embed_rev, ques_embed)
        #Match-LSTM Layer
        H_r = self.match_layer(H_p, H_p_rev, H_q)
        #Answer Pointer Layer
        dist = self.answer_layer(H_r)

        self.dist = dist
    def __add_loss_function(self):
        #TODO
    def __add_train_op(self):
        lr = self.lr
        ans = self.ans#(None, 2, pass_max_length)
        dist = self.dist#(None, 2, pass_max_length)

        loss = tf.reduce_mean(tf.cast(ans, tf.float32) * tf.log(dist) * (-1))

        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        self.train_op = train_op

    def __build(self):
        #add placeholders
        self.add_placeholder()
        #add variables / make architectures
        self.add_variables()
        #add predicted distribution
        self.add_predicted_dist()
        #add loss
        self.add_loss_function()
        #add train_op
        self.add_train_op()
