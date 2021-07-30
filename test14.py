import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

input = tf.random.normal([13, 19])
cell = tf.keras.layers.LSTMCell(7, implementation=1)
#rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
output = cell(inputs=input, states=[tf.zeros([13,7]), tf.zeros([13,7])])
print(output)