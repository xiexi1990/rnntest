import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.util import nest
import matplotlib.pyplot as plt

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
        self.Wd = self.add_weight(shape=(2, self.in_tanh_dim), initializer='glorot_uniform', name='Wd')
        self.bd = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros', name='bd')
        self.Ws = self.add_weight(shape=(3, self.in_tanh_dim), initializer='glorot_uniform', name='Ws')
        self.bs = self.add_weight(shape=(self.in_tanh_dim), initializer='zeros', name='bs')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.kernel_d = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform', name='kernel_d')
        self.kernel_s = self.add_weight(shape=(self.in_tanh_dim, self.units*4), initializer='glorot_uniform', name='kernel_s')
        self.kernel_c = self.add_weight(shape=(self.nclass, self.units * 4), initializer='glorot_uniform', name='kernel_c')
        self.bias = self.add_weight(shape=(self.units*4), initializer='zeros', name='bias')
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
        self.Wgmm = self.add_weight(shape=(input_dim, self.M * 5), initializer='glorot_uniform', name='Wgmm')
        self.bgmm = self.add_weight(shape=(self.M * 5), initializer='zeros', name='bgmm')
        self.Wsoftmax = self.add_weight(shape=(input_dim, 3), initializer='glorot_uniform', name='Wsoftmax')
        self.bsoftmax = self.add_weight(shape=(3), initializer='zeros', name='bsoftmax')
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
        return tf.concat([pi, mux, muy, sigmax, sigmay, p], axis=-1)
    def get_config(self):
        config = super(PostProcess, self).get_config()
        config.update({"M": self.M})
        return config

def N(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

units = 16
in_tanh_dim = 10
M = 8

def loss(y, pred):
    pi, mux, muy, sigmax, sigmay = tf.split(pred[:, :, :M * 5], 5, axis=-1)
    p = pred[:, :, M * 5:]
    xtp1 = tf.expand_dims(y[:, :, 0], axis=-1)
    ytp1 = tf.expand_dims(y[:, :, 1], axis=-1)
    stp1 = y[:, :, 2:5]
    w = tf.constant([1, 5, 100], dtype=tf.float32)
    lPd = tf.math.log(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
    lPs = tf.reduce_sum(w * stp1 * tf.math.log(p), axis=-1)
    return - (lPd + lPs)

def construct_model(rnn_cell_units, in_tanh_dim, nclass, stateful, M, batch_shape):
    rnn_cell = SGRUCell(units=rnn_cell_units, in_tanh_dim=in_tanh_dim, nclass=nclass, dropout=0., recurrent_dropout=0.)
    rnn_layer = tf.keras.layers.RNN(rnn_cell, return_state=False, return_sequences=True, stateful=stateful)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=batch_shape),
        rnn_layer,
        PostProcess(M=M)
      ])
    return model

def draw_real_char(ch):
    fig1 = plt.figure()
    drew = 0
    while drew <= 10:
        real_batch = take_batches.as_numpy_iterator().__next__()[0]
        for _i in range(np.size(real_batch, 0)):
            if real_batch[_i, 0, 5] == ch:
                drew += 1
                if drew > 10:
                    break
                real_plt = fig1.add_subplot((drew-1) // 5 + 1, 5, drew)
                print(drew, (drew-1) // 5 + 1)
                real_char = real_batch[_i, :, :]
                real_cur_x = 0
                real_cur_y = 0
                for _j in range(1, np.size(real_char, 0)):
                    real_next_x = real_cur_x + real_char[_j, 0]
                    real_next_y = real_cur_y + real_char[_j, 1]
                    if (real_char[_j, 2] == 1):
                        real_plt.plot([real_cur_x, real_next_x], [real_cur_y, real_next_y], color='black')
                    real_cur_x = real_next_x
                    real_cur_y = real_next_y

    plt.show()

def draw_chars(model, classes, maxlen):
    fig = plt.figure()
    ch_cnt = 1
    for ch in classes:
        ch_plt = fig.add_subplot(1, len(classes), ch_cnt)
        # real_plt = fig.add_subplot(2, len(classes), ch_cnt)
        # find = False
        # while not find:
        #     real_batch = take_batches.as_numpy_iterator().__next__()[0]
        #     for _i in range(np.size(real_batch, 0)):
        #         if real_batch[_i, 0, 5] == ch:
        #             find = True
        #             break
        # real_char = real_batch[_i, :, :]
        # real_cur_x = 0
        # real_cur_y = 0
        # for _j in range(1, np.size(real_char, 0)):
        #     real_next_x = real_cur_x + real_char[_j, 0]
        #     real_next_y = real_cur_y + real_char[_j, 1]
        #     if (real_char[_j, 2] == 1):
        #         real_plt.plot([real_cur_x, real_next_x], [real_cur_y, real_next_y], color='black')
        #     real_cur_x = real_next_x
        #     real_cur_y = real_next_y

        model.reset_states()
        pnt_in = np.array([[[0, 0, 0, 0, 0, ch]]])
        pnt_cnt = 0
        cur_x = 0
        cur_y = 0

        while pnt_cnt < maxlen:
            pred = np.squeeze(model(pnt_in))
            pi, mux, muy, sigmax, sigmay = np.split(pred[:M * 5], 5, axis=-1)
            p = pred[M * 5:]
            r1 = np.random.rand()
            sum = 0
            for i in range(M):
                sum += pi[i]
                if sum > r1:
                    [x_pred, y_pred] = np.random.multivariate_normal([mux[i], muy[i]], [[np.square(sigmax[i]), 0], [0, np.square(sigmay[i])]])
                    break
            r2 = np.random.rand()
            sum = 0
            s_pred = np.zeros(3)
            for i in range(3):
                sum += p[i]
                if sum > r2:
                    s_pred[i] = 1
       #             s_pred[np.argmax(p)] = 1
                    break


            next_x = cur_x + x_pred
            next_y = cur_y + y_pred
            if(s_pred[2] == 1):
                break
            if(s_pred[0] == 1):
                ch_plt.plot([cur_x, next_x], [cur_y, next_y], color='black')
            cur_x = next_x
            cur_y = next_y
            pnt_in[0, 0, 0] = x_pred
            pnt_in[0, 0, 1] = y_pred
            pnt_in[0, 0, 2:5] = s_pred
            pnt_cnt += 1
        ch_cnt += 1
    plt.show()

ckdir = 'gru'
if False:
    model = construct_model(units, in_tanh_dim, nclass, False, M, [None, None, 6])
    model.compile(loss=loss, optimizer=keras.optimizers.Adam())
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckdir + '/ck_{epoch}', save_weights_only=True)
    model.fit(take_batches, steps_per_epoch=30, epochs=10, callbacks=[checkpoint_callback])

else:
    # draw_real_char(5)
    # exit()
    model = construct_model(units, in_tanh_dim, nclass, True, M, [1, 1, 6])
    model.load_weights(tf.train.latest_checkpoint('test_weights'))
    # ii = 0
    # while True:
    #     a = take_batches.as_numpy_iterator().__next__()
    #     los = loss(a[1], model(a[0]))
    #     if(np.sum(los) < 0):
    #         break
    #     ii += 1
    # print(ii)
    # exit()




    # model = construct_model(units, in_tanh_dim, nclass, True, M, [1, 1, 6])
   # model.load_weights(tf.train.latest_checkpoint(ckdir))
  #  model.build(tf.TensorShape([1, 1, 6]))
    draw_chars(model, [5,6,7,8], 100)

exit()



# class CustomCallback(keras.callbacks.Callback):
#     def __init__(self, model):
#         self.model = model
#
#     def on_epoch_end(self, epoch):
#         y_pred = self.model.predict()
#         print('y predicted: ', y_pred)




