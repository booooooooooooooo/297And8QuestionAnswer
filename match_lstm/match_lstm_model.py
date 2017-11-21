import tensorflow as tf
import numpy as np

class LSTM_encoder:
    def __init__(self, cellName, input_size, state_size):
        self.cellName = cellName
        self.input_size = input_size
        self.state_size = state_size
    def add_variables(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.cellName):
            self.W_i = tf.get_variable("W_i", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_i = tf.get_variable("V_i", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_i = tf.get_variable("b_i", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_f = tf.get_variable("W_f", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_f = tf.get_variable("V_f", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_f = tf.get_variable("b_f", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_o = tf.get_variable("W_o", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_o = tf.get_variable("V_o", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_o = tf.get_variable("b_o", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)

            self.W_c = tf.get_variable("W_c", initializer = xavier_initializer, shape =  (self.input_size, self.state_size), dtype = tf.float32)
            self.V_c = tf.get_variable("V_c", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_c = tf.get_variable("b_c", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)
    def encode_one_step(self, batch_inputs, batch_states, batch_memories):
        i_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_i) + tf.matmul(batch_states, self.V_i) + self.b_i)
        f_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_f) + tf.matmul(batch_states, self.V_f) + self.b_f)
        o_k = tf.nn.sigmoid(tf.matmul(batch_inputs, self.W_o) + tf.matmul(batch_states, self.V_o) + self.b_o)
        new_batch_memories = f_k * batch_memories + i_k * tf.nn.tanh( tf.matmul(batch_inputs, self.W_c) + tf.matmul(batch_states, self.V_c) + self.b_c)
        new_batch_states = o_k * tf.nn.tanh(new_batch_memories)

        return new_batch_states, new_batch_memories
    def encode_sequence(self, batch_inputs_sequence, seq_len):
        # Assume all sequences have same length. Padding work is done in data related files.
        states = []
        batch_states = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        batch_memories = tf.zeros( ( tf.shape(batch_inputs_sequence)[0], self.state_size) ) #TODO: problem of None
        for i in xrange(seq_len):
            new_batch_states, new_batch_memories = self.encode_one_step(batch_inputs_sequence[:,i], batch_states, batch_memories)
            states.append(new_batch_states)
            batch_states, batch_memories = new_batch_states, new_batch_memories
        return tf.transpose( tf.stack(states) , (1, 0, 2))

def sanity_LSTM_encoder():
    input_size = 3
    state_size = 4
    seq_len = 5

    inputs_placeholder = tf.placeholder(tf.float32, shape=(None, seq_len, input_size))
    encoder = LSTM_encoder("test", input_size, state_size)
    encoder.add_variables()
    predicted = encoder.encode_sequence(inputs_placeholder, seq_len)


    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    inputs = np.zeros((10, seq_len, input_size))
    print sess.run(  tf.shape( sess.run(predicted, {inputs_placeholder : inputs}) )  )
    print sess.run(predicted, {inputs_placeholder : inputs})

if __name__ == "__main__":
    sanity_LSTM_encoder()
