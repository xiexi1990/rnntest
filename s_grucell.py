import pickle
import keras
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import AbstractRNNCell
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.util import nest
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

class SGRUCell(DropoutRNNCellMixin, AbstractRNNCell):
    def __init__(self, units, dropout=0., recurrent_dropout=0., **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    @property
    def state_size(self):
        return self.units
    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3), name='kernel', initializer=keras.initializers.glorot_uniform)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3), name='recurrent_kernel', initializer=keras.initializers.orthogonal)
        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, _train, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, _train, count=3)
        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs
        x_z = tf.matmul(inputs_z, self.kernel[:, :self.units])
        x_r = tf.matmul(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_h = tf.matmul(inputs_h, self.kernel[:, self.units * 2:])
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1
        recurrent_z = tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
        recurrent_r = tf.matmul(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
        z = tf.sigmoid(x_z + recurrent_z)
        r = tf.sigmoid(x_r + recurrent_r)
        recurrent_h = tf.matmul(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        hh = tf.tanh(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if nest.is_sequence(states) else h
        return h, new_state

    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({"units": self.units, 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout})
        return config

stacked_cell = tf.keras.layers.StackedRNNCells(
            [SGRUCell(units=16, dropout=0.2, recurrent_dropout=0.2) for _ in range(2)])
rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=False, return_sequences=True)

while True:
    a = take_batches.as_numpy_iterator().__next__()
    if np.size(a[0], 0) > 1:
        break

_train = False

d = rnn_layer(a[0])

model = keras.Sequential([
    keras.layers.Input(shape=(None, 6), dtype=tf.float32, ragged=False),
    keras.layers.Bidirectional(rnn_layer),
 #   keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(10, activation="softmax")),
])

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
model.summary(line_length=200)

_train = True

model.fit(take_batches, steps_per_epoch=10, epochs=10)

print("end")