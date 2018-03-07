import tensorflow as tf
class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, attentor, attentor_mask, input_size, state_size, style):
        '''
        input_size: size of encoded passage vector
        state_size = size of this cell's hidden units
        '''
        if style != "general" and style != "simple" and style != "gated":
            raise ValueError('MatchGRUCell style should be general, simple or gated')

        self.attentor = attentor
        self.attentor_mask = attentor_mask
        self.input_size = input_size
        self._state_size = state_size
        self.style = style

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
            _input_size = self.input_size#assume attentor and inputs has save vector length
            _state_size = self._state_size
            attentor = self.attentor
            attentor_mask = self.attentor_mask
            batch_size = tf.shape(attentor)[0]
            ques_max_length = tf.shape(attentor)[1]

            init = tf.contrib.layers.xavier_initializer()
            #attention
            W_q = tf.get_variable("W_q", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_p = tf.get_variable("W_p", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_r_0 = tf.get_variable("W_r_0", initializer = init, shape = (_state_size, _input_size), dtype = tf.float32)
            b_p = tf.get_variable("b_p", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            w = tf.get_variable("w", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)

            if self.style == "simple":
                G = tf.tanh(tf.matmul(attentor, tf.tile(tf.expand_dims(W_q, axis = [0]), [batch_size, 1, 1]))
                    + tf.expand_dims(tf.matmul(inputs, W_p) + tf.matmul(state, W_r_0) + b_p, [1]))#(batch_size, ques_max_length, _input_size)
            else:
                G = tf.tanh(tf.matmul(attentor, tf.tile(tf.expand_dims(W_q, axis = [0]), [batch_size, 1, 1]))
                    + tf.expand_dims(tf.matmul(inputs, W_p)  + b_p, [1]))#(batch_size, ques_max_length, _input_size)
            alpha = tf.reshape(tf.matmul(G, tf.tile(tf.reshape(w, [1, _input_size, 1]), [batch_size, 1, 1])), [batch_size, -1])#(batch_size, ques_max_length)
            #TODO:use 10e-1 to mask alpha before appling softmax ??
            alpha = tf.nn.softmax(alpha) * attentor_mask #(batch_size, ques_max_length)
            att_v = tf.reshape(tf.matmul(tf.expand_dims(alpha, [1]), attentor), [batch_size, _input_size])#(batch_size, _input_size)
            z = tf.concat([inputs, att_v], 1)#(batch_size, 2 * _input_size)

            if self.style == "gated":
                W_g = tf.get_variable("W_g", initializer = init, shape = (2 * _input_size, 2 * _input_size), dtype = tf.float32)
                g = tf.nn.sigmoid(tf.matmul(z, W_g))
                z = g * z
            #gru cell
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
