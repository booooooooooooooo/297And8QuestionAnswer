import tensorflow as tf

batch_size = 7
input_size = 5
state_size = 3
lstm_ans = tf.contrib.rnn.BasicLSTMCell(state_size)

hidden_state = tf.zeros(dtype = tf.float32, shape = [batch_size, state_size])
current_state = tf.zeros(dtype = tf.float32, shape = [batch_size, state_size])
state = hidden_state, current_state

lstm_input = tf.constant(1, dtype = tf.float32,  shape = (batch_size, input_size))
output, state = lstm_ans(lstm_input, state)
hidden_state, current_state = state

print output
print hidden_state
print current_state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(output)
    print sess.run(hidden_state)
    print sess.run(current_state)
