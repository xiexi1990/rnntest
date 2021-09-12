import pickle
import keras
import tensorflow as tf
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

class SGRUCell(AbstractRNNCell):
    def __init__(self, units, **kwargs):
        self.units = units
        super(SGRUCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states, training):
        h_tm1 = states
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs
        x_z = tf.matmul(inputs_z, self.kernel[:, :self.units])
        x_r = tf.matmul(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_h = tf.matmul(inputs_h, self.kernel[:, self.units * 2:])
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1
        recurrent_z = tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
        recurrent_r = tf.matmul(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)
        recurrent_h = tf.matmul(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        new_state = h
        return h, new_state

