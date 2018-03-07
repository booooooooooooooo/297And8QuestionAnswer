import tensorflow as tf

from Encoder import Encoder

class EncoderPreprocess(Encoder):
    def __init__(self, embed_matrix, passage, passage_mask, ques, ques_mask, state_size):
        self.embed_matrix = embed_matrix
        self.passage = passage
        self.passage_mask = passage_mask
        self.ques = ques
        self.ques_mask = ques_mask
        self.state_size = state_size

    def encode(self):
        '''
        return:
            H_p: (batch_size, pass_max_length, 2 * state_size)
            H_q: (batch_size, ques_max_length, 2 * state_size)
        '''
        with tf.variable_scope("embedding"):
            passage_embed = tf.nn.embedding_lookup(self.embed_matrix, self.passage)
            ques_embed = tf.nn.embedding_lookup(self.embed_matrix, self.ques)

        with tf.variable_scope("encode_preprocess"):
            lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(self.state_size)

            H_p_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, passage_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(self.passage_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_q_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(self.ques_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_p = tf.concat(H_p_pair, 2)
            H_q = tf.concat(H_q_pair, 2)
        return H_p, H_q
