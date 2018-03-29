import tensorflow as tf

from MatchGRUCell import MatchGRUCell
from Encoder import Encoder

class EncoderMatch(Encoder):
    def __init__(self, attentor, attentor_mask, target, target_mask, input_size, state_size, style):

        # if style != "general" and style != "simple" and style != "gated":
        #     raise ValueError('EncoderMatch does not support architecture:' + self.style)
        self.attentor = attentor
        self.attentor_mask = attentor_mask
        self.target = target
        self.target_mask = target_mask
        self.input_size = input_size# size of last dimension of attentor and target
        self.state_size = state_size
        self.style = style

    def encode(self):
        '''
        return
            H_r: (batch_size, pass_max_length, 2 * state_size)
        '''
        with tf.variable_scope("encode_match"):
            lstm_gru_fw = MatchGRUCell(self.attentor, self.attentor_mask, self.input_size, self.state_size, self.style)
            lstm_gru_bw = MatchGRUCell(self.attentor, self.attentor_mask, self.input_size, self.state_size, self.style)


            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                         lstm_gru_bw,
                                                         self.target,
                                                         sequence_length=tf.reduce_sum(tf.cast(self.target_mask, tf.int32), axis=1),
                                                         dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)

        return H_r
