import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

#input = tf.random.normal([13, 19])
input = tf.ones([7,9])
cell = tf.keras.layers.LSTMCell(4, implementation=1, kernel_initializer='ones')

stacked_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=4, implementation=1, kernel_initializer='ones', recurrent_initializer='ones') for _ in range(3)])
#rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=False, return_sequences=True)
#rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
output = stacked_cell(inputs=input,  states=[tf.zeros([7,4]), tf.zeros([7,4]), tf.zeros([7,4]), tf.zeros([7,4]), tf.zeros([7,4]), tf.zeros([7,4])])
print(output)