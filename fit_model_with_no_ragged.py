import pickle
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
from tensorflow.python.keras.layers import AbstractRNNCell


with open("x_y_100", "rb") as f:
    x, y = pickle.load(f)
    f.close()

xd = np.zeros(len(x), dtype=int)
for i in range(len(x)):
    xd[i] = np.size(x[i], 0)

uniquex = np.unique(xd)

def train_generator():
    i = 0
    while True:
        curlen = np.size(x[i], 0)
        xout = [x[i]]
        yout = [y[i]]
        i += 1
        if i == len(x):
            return
            i = 0
            continue
        while np.size(x[i], 0) == curlen:
            xout = np.append(xout, [x[i]], axis=0)
            yout = np.append(yout, [y[i]], axis=0)
            i += 1
            if i == len(x):
                return
                i = 0
                break
        yield (xout, np.expand_dims(yout, 1))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32),output_shapes=([None, None, 6], [None, 1, 10]))
take_batches = dataset.repeat().shuffle(1000)


# class SGRUCell(AbstractRNNCell):
#     def __init__(self, units, **kwargs):
#         self.units = units
#         super(SGRUCell, self).__init__(**kwargs)
#
#     @property
#     def state_size(self):
#         return self.units
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
#                                       initializer='uniform',
#                                       name='kernel')
#         self.recurrent_kernel = self.add_weight(
#             shape=(self.units, self.units),
#             initializer='uniform',
#             name='recurrent_kernel')
#         self.built = True
#
#     def call(self, inputs, states):
#         prev_output = states[0]
#         h = backend.dot(inputs, self.kernel)
#         output = h + backend.dot(prev_output, self.recurrent_kernel)
#         return output, output

class S_GRU(keras.layers.Layer):
    def __init__(self, h_size, nlayer, **kwargs):
        super().__init__(**kwargs)
        self.h_size = h_size
        self.nlayer = nlayer

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.nlayer, 3, self.h_size, self.h_size),initializer=keras.initializers.glorot_uniform)
        self.U = self.add_weight(shape=(self.nlayer, 3, input_shape[2], self.h_size), initializer=keras.initializers.glorot_uniform)

    def call(self, inputs, **kwargs):
        in_shape = tf.shape(inputs)
        batch_size = in_shape[0]
        init_st = tf.zeros([self.nlayer, batch_size, self.h_size])

        def time_step(prev, xt):
            st = []
            inp = xt
            for i in range(self.nlayer):
                z = tf.sigmoid(tf.matmul(inp, self.U[i][0]) + tf.matmul(prev[i], self.W[i][0]))
                r = tf.sigmoid(tf.matmul(inp, self.U[i][1]) + tf.matmul(prev[i], self.W[i][1]))
                h = tf.tanh(tf.matmul(inp, self.U[i][2]) + tf.matmul(r * prev[i], self.W[i][2]))
                st_i = (1 - z) * h + z * prev[i]
                inp = st_i
                st.append(st_i)
            return tf.stack(st)

        outputs = tf.scan(time_step, tf.transpose(inputs, [1, 0, 2]), initializer=init_st)
        return tf.transpose(outputs[:, self.nlayer - 1, :, :], [1, 0, 2])

class S_LSTM(keras.layers.Layer):
    def __init__(self, h_size, nlayer, **kwargs):
        super(S_LSTM, self).__init__(**kwargs)
        self.h_size = h_size
        self.nlayer = nlayer
        self.Wxi, self.Wxf, self.Wxc, self.Wxo = [], [], [], []
        self.Whi, self.Whf, self.Whc, self.Who = [], [], [], []
        self.Whsi, self.Whsf, self.Whsc, self.Whso = [], [], [], []
        self.Wc = []
        self.bi, self.bf, self.bc, self.bo = [], [], [], []

    def build(self, input_shape):
        for i in range(self.nlayer):
            self.Wxi.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Wxf.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Wxc.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Wxo.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer=keras.initializers.glorot_uniform))
            if i > 0:
                self.Whsi.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
                self.Whsf.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
                self.Whsc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
                self.Whso.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Whi.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Whf.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Whc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Who.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.Wc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer=keras.initializers.glorot_uniform))
            self.bi.append(self.add_weight(shape=(self.h_size,), initializer=keras.initializers.zeros))
            self.bf.append(self.add_weight(shape=(self.h_size,), initializer=keras.initializers.zeros))
            self.bc.append(self.add_weight(shape=(self.h_size,), initializer=keras.initializers.zeros))
            self.bo.append(self.add_weight(shape=(self.h_size,), initializer=keras.initializers.zeros))
         #   self.h.append(tf.Variable(tf.zeros([1, self.h_size]), shape=[None, self.h_size]))
         #   self.c.append(tf.Variable(tf.zeros([1, self.h_size]), shape=[None, self.h_size]))


    def call(self, inputs, **kwargs):
        in_shape = tf.shape(inputs)
        batch_size = in_shape[0]
        init_c = []
        init_h = []
        if _train:
            print("training")
        else:
            print("not train")

        for i in range(self.nlayer):
            init_c.append(tf.zeros([batch_size, self.h_size]))
            init_h.append(tf.zeros([batch_size, self.h_size]))

        def time_step(prev, xt):
            c_1, h_1 = tf.unstack(prev)
            c, h = [], []
            for i in range(self.nlayer):
                _i = tf.math.sigmoid(tf.matmul(xt, self.Wxi[i]) + (i > 0 and tf.matmul(h[i - 1], self.Whsi[i - 1])) + tf.matmul(h_1[i], self.Whi[i]) + self.bi[i])
                _f = tf.math.sigmoid(tf.matmul(xt, self.Wxf[i]) + (i > 0 and tf.matmul(h[i - 1], self.Whsf[i - 1])) + tf.matmul(h_1[i], self.Whf[i]) + self.bf[i])
                _o = tf.math.sigmoid(tf.matmul(xt, self.Wxo[i]) + (i > 0 and tf.matmul(h[i - 1], self.Whso[i - 1])) + tf.matmul(h_1[i], self.Who[i]) + self.bo[i])
                _c =    tf.math.tanh(tf.matmul(xt, self.Wxc[i]) + (i > 0 and tf.matmul(h[i - 1], self.Whsc[i - 1])) + tf.matmul(h_1[i], self.Whc[i]) + self.bc[i])
                c.append(tf.multiply(_f, c_1[i]) + tf.multiply(_i, _c))
                h.append(tf.multiply(_o, tf.tanh(c[i])))
            return tf.stack([c, h])


        outputs = tf.scan(time_step, tf.transpose(inputs, [1, 0, 2]), tf.stack([init_c, init_h]))
        return tf.transpose(outputs[:, 1, self.nlayer-1, :, :], [1, 0, 2])

    def get_config(self):
        config = super(S_LSTM, self).get_config()
        config.update({"h_size": self.h_size, "nlayer": self.nlayer})
        return config


_train = False

stacked_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=16, implementation=1) for _ in range(2)])
#rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=False, return_sequences=True)
rnn_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units=16, implementation=1), return_state=False, return_sequences=True)

# stacked_gru_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=16, implementation=1) for _ in range(2)])
# gru_layer = tf.keras.layers.RNN(stacked_gru_cell, return_state=False, return_sequences=True)
gru_layer2 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(units=16,implementation=1))

s_lstm_layer = S_LSTM(16, 2)
# s_gru_layer = S_GRU(16, 2)

while True:
    a = take_batches.as_numpy_iterator().__next__()
    if np.size(a[0], 0) > 1:
        break

#f = gru_layer(a[0])
g = gru_layer2(a[0])
c = s_lstm_layer(a[0])
d = rnn_layer(a[0])
#e = s_gru_layer(a[0])

model = keras.Sequential([
    keras.layers.Input(shape=(None, 6), dtype=tf.float32, ragged=False),
    #keras.layers.Bidirectional(rnn_layer),
    rnn_layer,
 #   keras.layers.Bidirectional(s_lstm_layer),
 #   s_gru_layer,
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(10, activation="softmax")),

])

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

model.summary(line_length=200)

_train = True

model.fit(take_batches, steps_per_epoch=10, epochs=3)

print("end")