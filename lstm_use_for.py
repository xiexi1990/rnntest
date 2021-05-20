import tensorflow as tf
import numpy as np
import sys

batch_size = 1
hidden_size = 4
num_steps = 3
input_dim = 5

np.random.seed(123)
#input = np.random.randn(batch_size, num_steps, input_dim)
input = np.ones([batch_size, num_steps, input_dim], dtype=int)

#input[1, 6:] = 0
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_steps, input_dim], name='input_x')
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs = []
with tf.variable_scope('RNN', initializer= tf.ones_initializer):
    for i in range(num_steps):
      #  if i > 0:
            # print(tf.get_variable_scope())
      #      tf.get_variable_scope().reuse_variables()

        output = lstm_cell(x[:, i, :], initial_state)

        outputs.append(output)
        if i > 0:
            outputs[i]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    np.set_printoptions(threshold=sys.maxsize)

    result = sess.run(outputs, feed_dict={x: input})
    print(np.asarray(result))

