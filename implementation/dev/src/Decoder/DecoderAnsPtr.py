import tensorflow as tf

from Decoder import Decoder

class DecoderAnsPtr(Decoder):
    def __init__(self, H_r, passage_mask, input_size):
        self.H_r = H_r
        self.passage_mask = passage_mask
        self.input_size = input_size
    def decode(self):
        '''
        paras
            H_r:        (batch_size, pass_max_length, input_size)
        return

        '''
        batch_size = tf.shape(self.H_r)[0]
        # passage_max_length = tf.shape(H_r)[1]
        input_size = self.input_size

        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decode_ans_ptr"):
            V = tf.get_variable("V", shape = (input_size, input_size), dtype = tf.float32, initializer = init)
            W_a = tf.get_variable("W_a", shape = (input_size, input_size), dtype = tf.float32, initializer = init)
            b_a = tf.get_variable("b_a", shape = (input_size, ), dtype = tf.float32, initializer = init)
            v = tf.get_variable("v", shape = (input_size, ), dtype = tf.float32, initializer = init)
            c = tf.get_variable("c", shape = (), dtype = tf.float32, initializer = init)


            F_s = tf.tanh( tf.matmul(self.H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                           + b_a)#(batch_size, pass_max_length, num_units)
            beta_s = tf.squeeze(tf.matmul(F_s,
                                          tf.tile(tf.reshape(v, [1, -1, 1]),
                                                  [batch_size, 1, 1])))#(batch_size, pass_max_length)





            prob_s = tf.nn.softmax(beta_s) * self.passage_mask#(batch_size, pass_max_length)
            h_a = tf.squeeze(tf.matmul(tf.expand_dims(prob_s, [1]), self.H_r))#(batch_size, 2 * num_units)
            F_e = tf.tanh( tf.matmul(self.H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                        +  tf.expand_dims(tf.matmul(h_a, W_a) + b_a , [1]))#(batch_size, pass_max_length, num_units)
            beta_e = tf.squeeze(tf.matmul(F_e, tf.tile(tf.reshape(v, [1, -1, 1]), [batch_size, 1, 1])))#(batch_size, pass_max_length)




        return beta_s, beta_e
