import tensorflow as tf
import numpy as np
batch_size = 2
hidden_size = 4
num_steps = 3
input_dim = 5
np.random.seed(123)
input = np.ones([batch_size, num_steps, input_dim], dtype=int)
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_steps, input_dim], name='input_x')
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
z_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs = []

with tf.variable_scope('for_loop', initializer= tf.ones_initializer):
    #out, stat = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=z_state)
    #out2,stat2 = tf.nn.dynamic_rnn(lstm_cell,x[:,1,:], initial_state=stat)

 #   output = lstm_cell(x[:, 0, :], z_state)

 #   output2 = lstm_cell(x[:, 1, :], output[1])

    for i in range(num_steps):

        if i > 0:
         #   tf.get_variable_scope().reuse_variables()
            output = lstm_cell(x[:, i, :], outputs[i-1][1])
        else:
            output = lstm_cell(x[:, i, :], z_state)
        outputs.append(output)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    result = sess.run(outputs, feed_dict={x: input})
    print(result)

