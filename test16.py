import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.util import nest



nclass = 30
with open("x_y_lb_" + str(nclass), "rb") as f:
    x, y = pickle.load(f)

dataset = tf.data.Dataset.from_tensor_slices((x, y))
take_batches = dataset.repeat().shuffle(1000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

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
            [SGRUCell(units=16, dropout=0., recurrent_dropout=0.) for _ in range(2)])
rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=False, return_sequences=True)

_train = False

a = take_batches.as_numpy_iterator().__next__()
print(a)

d = rnn_layer(a[0])