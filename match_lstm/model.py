'''
Used by tune.py to train better tensor graph.
'''
import tensorflow as tf

'''
TODO:
use BasicLSTMCell professionally
all calculation only use embed_s, batch_s and num_units
padding
give nice names to important tensors

regularization
gradient decent clipping

efficient training: nce etc.

mark public and private def
include embed_s, batch_s in title of saved model
include num_units, lr, n_epoch in title of saved model too
'''
class Model:
    def __init__(self, config):
        self.config = config
    def add_placeholder(self):
        batch_s = self.config.batch_s#trainning and test have same batch_s

        self.ques = tf.placeholder(tf.int32, shape = (batch_s, None), name = "ques_ph")#(batch_s, ques_l)
        self.ques_mask = tf.placeholder(tf.int32, shape = (batch_s, ), name = "ques_pad_ph")#(batch_s, ques_l)
        self.passage = tf.placeholder(tf.int32, shape = (batch_s, None), name = "pass_ph")#(batch_s, pass_l)
        self.passage_mask = tf.placeholder(tf.int32, shape = (batch_s, ), name = "pass_pad_ph")#(batch_s, pass_l)
        self.ans = tf.placeholder(tf.int32, shape = (batch_s, 2), name = "ans_ph")#(batch_s, 2)
    def add_variables(self):
        embed_s = self.config.embed_s
        num_units = self.config.num_units

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
    def embed_layer(self):
        embed_matrix = self.config.embed_matrix
        passage = self.passage
        ques = self.ques

        pass_embed = tf.nn.embedding_lookup(embed_matrix, passage)#(batch_s, pass_l, embed_s)
        ques_embed = tf.nn.embedding_lookup(embed_matrix, ques)#(batch_s, ques_l, embed_s)
        return pass_embed, ques_embed
    def pre_layer(self, pass_embed, ques_embed):
        '''
        paras:
            pass_embed: (batch_s, pass_l, embed_s)
            ques_embed: (batch_s, ques_l, embed_s)
        return:
            H_p: (batch_s, pass_l, num_units)
            H_q: (batch_s, ques_l, num_units)
        '''
        batch_s = self.config.batch_s
        embed_s = self.config.embed_s
        num_units = self.config.num_units

        lstm_pre = self.lstm_pre

        passage_mask = self.passage_mask
        ques_mask = self.ques_mask

        initial_state = lstm_pre.zero_state(batch_s, dtype=tf.float32)

        H_p, _ = tf.nn.dynamic_rnn(lstm_pre, pass_embed, sequence_length = passage_mask,
                                           initial_state=initial_state,
                                           dtype=tf.float32)
        H_q, _ = tf.nn.dynamic_rnn(lstm_pre, ques_embed, sequence_length = ques_mask,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        return H_p, H_q

    # def match_attention(self, H_q, h_p, h_r):
    #     '''
    #     paras:
    #         H_q: (batch_s, ques_l, num_units)
    #         h_p: (batch_s, num_units)
    #         h_r: (batch_s, num_units)
    #     return:
    #         z: (batch_s, 2 * num_units)
    #     '''
    #     ques_l = self.config.ques_l
    #     num_units = self.config.num_units
    #     W_q = self.W_q
    #     W_p = self.W_p
    #     W_r = self.W_r
    #     b_p = self.b_p
    #     w = self.w
    #     b = self.b
    #
    #     G_part1 = abc_mul_cd(H_q, W_q, ques_l, num_units, num_units)#(batch_s, ques_l, num_units)
    #     G_part2 = tf.matmul(h_p, W_p) + tf.matmul(h_r, W_r) + b_p #(batch_s, num_units)
    #     G_sum = abc_plus_ac(G_part1, G_part2, ques_l, num_units)#(batch_s, ques_l, num_units)
    #     G = tf.nn.tanh( G_sum )# batch_s * ques_l * l
    #
    #     alpha = tf.nn.softmax( abc_mul_c(G, w, ques_l, num_units) + b )#(batch_s, ques_l)
    #
    #     att_v = tf.matmul( tf.transpose(H_q, (0,2,1)), tf.reshape(alpha, (-1, ques_l, 1)) )#(batch_s, num_units, 1)
    #     att_v = tf.reshape(att_v, (-1, num_units))#(batch_s, num_units)
    #
    #     z = tf.concat(1, [h_p, att_v])#(batch_s, 2 * num_units)
    #
    #     return z
    # def match_layer(self, H_p, H_q):
    #     '''
    #     paras
    #         H_p:        (None, pass_l, num_units)
    #         H_q:        (None, ques_l, num_units)
    #     return
    #         H_r:        (None, pass_l, num_units)
    #     '''
    #     lstm = self.lstm_match
    #     num_units = self.config.num_units
    #     batch_s = self.config.batch_s
    #
    #     h_p_lst = tf.unstack(H_p, axis = 1)
    #
    #     h_r_lst = []
    #     h_r = lstm.zero_state(batch_s, tf.float32)
    #     for i in xrange(pass_len):
    #         h_p = h_p_lst[i]
    #         z_i = match_attention(H_q, h_p, h_r)
    #         h_r = lstm(z_i, h_r)
    #         h_r_lst.append(h_r)
    #     H_r = tf.stack(h_r_lst)# (pass_len, None, num_units)
    #
    #     return H_r
    # def answer_attention(self, H_r, h_a):
    #     '''
    #     paras:
    #         H_r: (batch_s, pass_l, 2 * num_units)
    #         h_a: (batch_s, num_units)
    #     return:
    #         beta: (batch_s, pass_l)
    #         input_lstm: (batch_s, 2 * num_units)
    #
    #     '''
    #     V = self.V
    #     pass_l = self.config.pass_l
    #     num_units = self.config.num_units
    #     W_a = self.W_a
    #     b_a = self.b_a
    #     v = self.v
    #     c = self.c
    #
    #     F_part1 = abc_mul_cd(H_r, V, pass_l, 2 * num_units, num_units)# (batch_s, pass_l, num_units)
    #     F_part2 = tf.matmul(h_a, W_a) + b_a # (batch_s, num_units)
    #     F_sum = abc_plus_ac(F_part1, F_part2, pass_l, num_units )
    #     F = tf.nn.tanh( F_sum )# (batch_s, pass_l, num_units)
    #
    #     beta = tf.nn.softmax( abc_mul_c(F, v, pass_l, num_units) + c )#(batch_s, pass_l)
    #
    #     input_lstm = tf.matmul( tf.transpose(H_r, (0,2,1)), tf.reshape(beta, (-1, pass_l, 1)) )#(batch_s, 2 * num_units, 1)
    #     input_lstm = tf.reshape(input_lstm, (-1, 2 * num_units))
    #
    #     return beta, input_lstm
    # def answer_layer(self, H_r):
    #     '''
    #     paras
    #         H_r:        (None, pass_l, 2 * num_units)
    #     return
    #         dist:       (None, 2, pass_l)
    #     '''
    #     lstm = self.lstm_ans
    #     num_units = self.config.num_units
    #     batch_s = self.config.batch_s
    #
    #     dist_lst = []
    #     h_a = lstm.zero_state(batch_s, tf.float32)
    #     for i in xrange(2):
    #         beta, input_lstm = match_attention(H_r, h_a)
    #         h_a = tf.matmul(input_lstm, U_ans)
    #         dist_lst.append(beta)
    #     dist = tf.stack(dist_lst)#(2, None, pass_l)
    #
    #     return tf.transpose(dist, (1,0,2))#(None, 2, pass_l)
    # def add_predicted_dist(self):
    #     #embdding
    #     pass_embed, ques_embed = self.embed_layer()
    #     #Preprocessing Layer
    #     H_p, H_q = pre_layer(pass_embed, ques_embed)
    #     #Match-LSTM Layer
    #     H_r_r = match_layer(H_p, H_q)
    #     H_r_l = match_layer(H_p, H_q) #TODO: bidirectional lstm and padding
    #     H_r = tf.concat(2, [H_r_r, H_r_l]) ## (pass_len, None, 2 * l)
    #     H_r = tf.transpose(H_r, (1, 0, 2))
    #     #Answer Pointer Layer
    #     self.dist = answer_layer(H_p)
    #
    # def add_train_op(self):
    #     #TODO
    #     loss = tf.reduce_mean( )
    # def build(self):
    #     #add placeholders
    #     add_placeholder()
    #     #add variables / make architectures
    #     add_variables()
    #     #get predicted distribution
    #     add_predicted_dist()
    #     #add train_op
    #     add_train_op()
    #
    # def train_batch(self, sess, batch_data):
    #
    # def train_epoch(self, sess):
    #
    # def fit(self, sess):
