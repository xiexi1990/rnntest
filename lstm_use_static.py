import tensorflow as tf
import numpy as np
batch_size = 1
hidden_size = 4
num_steps = 2
input_dim = 5
np.random.seed(123)
input = np.ones([batch_size, num_steps, input_dim], dtype=int)


T = 2
rnn_state_size = 4
num_layers = 2
x = tf.placeholder(dtype=tf.float32, shape=[None, T, 3])
stacked_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_state_size) for _ in range(num_layers)])
init_state = stacked_cell.zero_state(batch_size, dtype=tf.float32)
output_list, final_state = tf.nn.dynamic_rnn(stacked_cell, x, initial_state=init_state)


# x = tf.placeholder(dtype=tf.float32, shape=[None, num_steps, input_dim], name='input_x')
# lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for _ in range(2)])
# initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
# y = tf.unstack(x, axis=1)
# with tf.variable_scope('static_rnn', initializer= tf.ones_initializer):
#     output, state = tf.nn.dynamic_rnn(lstm_cell, x,  initial_state=initial_state)
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    result = (sess.run([output, state], feed_dict={x: input}))
    print(result)
