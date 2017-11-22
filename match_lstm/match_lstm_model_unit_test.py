import tensorflow as tf
import numpy as np

from match_lstm_model import *

def sanity_LSTM_encoder():
    input_size = 3
    state_size = 4
    seq_len = 5

    inputs_placeholder = tf.placeholder(tf.float32, shape=(None, seq_len, input_size))
    encoder = LSTM_encoder("sanity", input_size, state_size)
    encoder.add_variables()
    predicted = encoder.encode_sequence(inputs_placeholder, seq_len)


    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    inputs = np.zeros((10, seq_len, input_size))
    print sess.run(  tf.shape( sess.run(predicted, {inputs_placeholder : inputs}) )  )
    print sess.run(predicted, {inputs_placeholder : inputs})
def sanity_Attention_match():
    batch_size  = 10
    question_len = 5
    state_size = 3

    att_match = Attention_match("sanity", state_size)
    att_match.add_variables()
    H_q_placeholder = tf.placeholder(tf.float32, shape = (None, question_len, state_size))
    h_p_placeholder = tf.placeholder(tf.float32, shape = (None,  state_size))
    h_r_placeholder = tf.placeholder(tf.float32, shape = (None,  state_size))
    predicted = att_match.attention_one_step(H_q_placeholder, question_len, h_p_placeholder, h_r_placeholder)

    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    H_q = np.zeros((batch_size, question_len, state_size))
    h_p = np.zeros( (batch_size, state_size) )
    h_r = np.zeros( (batch_size, state_size) )
    result = sess.run(predicted, { H_q_placeholder : H_q, h_p_placeholder : h_p, h_r_placeholder : h_r })
    print result
    print sess.run( tf.shape(result) )

def sanity_Attention_answer():
    batch_size  = 10
    question_len = 5
    state_size = 3

    att_ans = Attention_ans("sanity", state_size)
    att_ans.add_variables()
    H_r_hat_placeholder = tf.placeholder(tf.float32, shape = (None, question_len, state_size * 2))
    h_a_placeholder = tf.placeholder(tf.float32, shape = (None,  state_size))
    predicted = att_ans.attention_one_step(H_r_hat_placeholder, question_len, h_a_placeholder)

    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )
    H_r_hat = np.zeros((batch_size, question_len, state_size * 2))
    h_a = np.zeros( (batch_size, state_size) )
    beta, state = sess.run(predicted, { H_r_hat_placeholder : H_r_hat, h_a_placeholder : h_a })
    print beta
    print state
    print sess.run( tf.shape(beta) )
    print sess.run( tf.shape(state) )
def test_tensorflow():
    a = tf.zeros((3,4))
    b = tf.zeros((3,4))
    print tf.concat(1, [a,b])
if __name__ == "__main__":
    # sanity_LSTM_encoder()
    # sanity_Attention_match()
    sanity_Attention_answer()
    # test_tensorflow()
