import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.util import nest

tf.random.set_seed(123)
np.random.seed(1234)

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

units = 16
in_tan_dim = 10
M = 8

class SGRUCell(DropoutRNNCellMixin, keras.layers.Layer):
    def __init__(self, units, in_tanh_dim, nclass, dropout=0., recurrent_dropout=0., **kwargs):
        super(SGRUCell, self).__init__(**kwargs)
        self.units = units
        self.in_tanh_dim = in_tanh_dim
        self.nclass = nclass
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
        assert len(input_shape) == 2 and input_dim == 6

        self.Wd = self.add_weight(shape=(2, self.in_tanh_dim), initializer='glorot_uniform')
        self.bd = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros')
        self.Ws = self.add_weight(shape=(3, self.in_tanh_dim), initializer='glorot_uniform')
        self.bs = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros')

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal')
        self.kernel_d = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform')
        self.kernel_s = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform')
        self.kernel_c = self.add_weight(shape=(self.nclass, self.units * 4), initializer='glorot_uniform')
        self.bias = self.add_weight(shape=(self.units*4), initializer='zeros')

        self.built = True

    def call(self, inputs, states, training):
        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        d = inputs[:, 0:2]
        s = inputs[:, 2:5]
        ch = tf.cast(inputs[:, 5], tf.int32)
        _d = tf.tanh(tf.matmul(d, self.Wd) + self.bd)
        _s = tf.tanh(tf.matmul(s, self.Ws) + self.bs)
        _ch = tf.one_hot(ch, self.nclass)
        _d_mask = self.get_dropout_mask_for_cell(_d, training, count=3)
        _s_mask = self.get_dropout_mask_for_cell(_s, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        if 0. < self.dropout < 1.:
            _d_z = _d * _d_mask[0]
            _d_r = _d * _d_mask[1]
            _d_h = _d * _d_mask[2]
            _s_z = _s * _s_mask[0]
            _s_r = _s * _s_mask[1]
            _s_h = _s * _s_mask[2]
        else:
            _d_z = _d
            _d_r = _d
            _d_h = _d
            _s_z = _s
            _s_r = _s
            _s_h = _s
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1
        z = tf.sigmoid(tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
                       + tf.matmul(_d_z, self.kernel_d[:, :self.units])
                       + tf.matmul(_s_z, self.kernel_s[:, :self.units])
                       + tf.matmul(_ch, self.kernel_c[:, :self.units])
                       + self.bias[:self.units])
        r = tf.sigmoid(tf.matmul(h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
                       + tf.matmul(_d_r, self.kernel_d[:, self.units:self.units * 2])
                       + tf.matmul(_s_r, self.kernel_s[:, self.units:self.units * 2])
                       + tf.matmul(_ch, self.kernel_c[:, self.units:self.units * 2])
                       + self.bias[self.units:self.units * 2])
        hh = tf.tanh(tf.matmul(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:self.units * 3])
                       + tf.matmul(_d_h, self.kernel_d[:, self.units * 2:self.units * 3])
                       + tf.matmul(_s_h, self.kernel_s[:, self.units * 2:self.units * 3])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 2:self.units * 3])
                       + self.bias[self.units * 2:self.units * 3])
        h = z * h_tm1 + (1 - z) * hh
        o = tf.tanh(tf.matmul(h, self.recurrent_kernel[:, self.units * 3:])
                       + tf.matmul(_d_r, self.kernel_d[:, self.units * 3:])
                       + tf.matmul(_s_r, self.kernel_s[:, self.units * 3:])
                       + tf.matmul(_ch, self.kernel_c[:, self.units * 3:])
                       + self.bias[self.units * 3:])
        new_state = [h] if nest.is_sequence(states) else h
        return o, new_state


    def get_config(self):
        config = super(SGRUCell, self).get_config()
        config.update({'units': self.units, 'in_tanh_dim': self.in_tanh_dim, 'nclass': self.nclass, 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout})
        return config

class PostProcess(keras.layers.Layer):
    def __init__(self, M, **kwargs):
        super(PostProcess, self).__init__(**kwargs)
        self.M = M

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wgmm = self.add_weight(shape=(input_dim, self.M * 5), initializer='glorot_uniform')
        self.bgmm = self.add_weight(shape=(self.M * 5), initializer='zeros')
        self.Wsoftmax = self.add_weight(shape=(input_dim, 3), initializer='glorot_uniform')
        self.bsoftmax = self.add_weight(shape=(3), initializer='zeros')
        self.built = True


    def call(self, inputs, **kwargs):
        R5M = tf.matmul(inputs, self.Wgmm) + self.bgmm
        _pi = R5M[:, :, :self.M]
        pi = tf.exp(_pi) / tf.reduce_sum(tf.exp(_pi), axis=-1, keepdims=True)
        mux = R5M[:, :, self.M:self.M * 2]
        muy = R5M[:, :, self.M * 2:self.M * 3]
        sigmax = tf.exp(R5M[:, :, self.M * 3:self.M * 4])
        sigmay = tf.exp(R5M[:, :, self.M * 4:])

        R3 = tf.matmul(inputs, self.Wsoftmax) + self.bsoftmax
        p = tf.exp(R3) / tf.reduce_sum(tf.exp(R3), axis=-1, keepdims=True)

        return pi, mux, muy, sigmax, sigmay, p

    def get_config(self):
        config = super(PostProcess, self).get_config()
        config.update({"M": self.M})
        return config

def N(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

rnn_cell = SGRUCell(units=units, in_tanh_dim=in_tan_dim, nclass=nclass, dropout=0., recurrent_dropout=0.)
rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True)
post_process_layer = PostProcess(M=M)
#
# print(tf.executing_eagerly())
#

#
# a = take_batches.as_numpy_iterator().__next__()
# d = rnn_layer(a[0])
# loss = keras.losses.mean_squared_error(a[1], tf.matmul(d, tf.ones((16, 6))))
# pi, mux, muy, sigmax, sigmay, p = post_process_layer(d)
#
# y = a[1]
# xtp1 = tf.expand_dims(y[:,:,0], axis=-1)
# ytp1 = tf.expand_dims(y[:,:,1], axis=-1)
# stp1 = y[:,:,2:5]
#
# w = tf.constant([1,5,100], dtype=tf.float32)
#
# lPd = tf.math.log(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
# lPs= tf.reduce_sum(w * stp1 * tf.math.log(p), axis=-1)
#
# loss = - tf.reduce_sum(lPd + lPs, axis=-1)
#
#
#exit()


class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pi, mux, muy, sigmax, sigmay, p = self(x, training=True)  # Forward pass
            xtp1 = tf.expand_dims(y[:,:,0], axis=-1)
            ytp1 = tf.expand_dims(y[:,:,1], axis=-1)
            stp1 = y[:,:,2:5]
            w = tf.constant([1,5,100], dtype=tf.float32)
            lPd = tf.math.log(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
            lPs= tf.reduce_sum(w * stp1 * tf.math.log(p), axis=-1)
            loss = - (lPd + lPs)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
      #  mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]

# Construct an instance of CustomModel
inputs = keras.Input(shape=(None,6))
gru_out = rnn_layer(inputs)
outputs = post_process_layer(gru_out)
model = CustomModel(inputs, outputs)

# We don't passs a loss or metrics here.
model.compile(optimizer="adam")

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch):
        y_pred = self.model.predict()
        print('y predicted: ', y_pred)



model.fit(take_batches, steps_per_epoch=100, epochs=100, callbacks=[CustomCallback(model)])

