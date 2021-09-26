import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.util import nest

nclass = 10
repeat = 10
with open("x_y_lb_" + str(nclass) + "_repeat_" + str(repeat), "rb") as f:
    x, y = pickle.load(f)

dataset = tf.data.Dataset.from_generator(lambda: iter(zip(x, y)), output_types=(tf.float32, tf.float32),output_shapes=([None, None, 6], [None, None, 6]))
take_batches = dataset.repeat().shuffle(1000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

class SGRUCell(DropoutRNNCellMixin, AbstractRNNCell):
    def __init__(self, units, in_tanh_dim, ch_class, dropout=0., recurrent_dropout=0., **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.in_tanh_dim = in_tanh_dim
        self.ch_class = ch_class
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    @property
    def state_size(self):
        return self.units
    @property
    def output_size(self):
        #return self.units
        return 30

    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert input_dim == 6

        self.Wd = self.add_weight(shape=(2, self.in_tanh_dim), initializer='glorot_uniform')
        self.bd = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros')
        self.Ws = self.add_weight(shape=(3, self.in_tanh_dim), initializer='glorot_uniform')
        self.bs = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros')

        self.kernel_h = self.add_weight(shape=(self.units, self.units*4), initializer='orthogonal')
        self.kernel_d = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform')
        self.kernel_s = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform')
        self.kernel_c = self.add_weight(shape=(self.ch_class, self.units * 4), initializer='glorot_uniform')
        self.bias = self.add_weight(shape=(self.units*4), initializer='zeros')

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

#        return h, new_state
        return tf.matmul(h, tf.Variable(np.random.normal(size=(16,30)), dtype=float)), new_state

    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({"units": self.units, 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout})
        return config

#stacked_cell = tf.keras.layers.StackedRNNCells([SGRUCell(units=16, in_tanh_dim=10, dropout=0., recurrent_dropout=0.) for _ in range(2)])
rnn_cell = SGRUCell(units=16, in_tanh_dim=10, dropout=0., recurrent_dropout=0.)
rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True)

_train = False

a = take_batches.as_numpy_iterator().__next__()


d = rnn_layer(a[0])

print(d)