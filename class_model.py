import tensorflow as tf
import keras
import numpy as np


class Model(keras.Model):

    def __init__(self, rnn_state_size, numlayers, M, b, T, learning_rate):
        super().__init__()
        print("model init")
        self.rnn_state_size = rnn_state_size
        self.numlayers = numlayers
        self.M = M
        self.b = b
        self.T = T
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        print("model build")
        stacked_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=self.rnn_state_size) for _ in range(self.numlayers)])
        self.rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=True)

        NOUT = 1 + self.M * 6
        self.output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.rnn_state_size, NOUT]))
        self.output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[NOUT]))

    def call(self, inputs, training=None, mask=None):
        print("model call")
        output_list, final_state = self.rnn_layer(inputs)
        _concat_output_list = tf.concat(output_list, 1)
        _reshape_concat_output_list = tf.reshape(_concat_output_list, [-1, self.rnn_state_size])
        output = _reshape_concat_output_list * self.output_w + self.output_b
        return output

    def train_step(self, data):
        @tf.function
        def expand(x, dim, N):
            _expand = tf.expand_dims(x, dim)
            _one_hot = tf.squeeze(tf.one_hot(indices=[dim], depth=tf.rank(_expand), on_value=N, off_value=1))
            return tf.tile(_expand, _one_hot)

        @tf.function
        def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z = tf.math.square((x1 - mu1) / sigma1) + tf.math.square((x2 - mu2) / sigma2) \
                - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            return tf.math.exp(-z / (2 * (1 - tf.math.square(rho)))) / \
                   (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.math.square(rho)))

        print("model train step")

        x, y = data

        _reshape_y = tf.reshape(y, [-1, 3])
        y1, y2, y_end_of_stroke = tf.unstack(_reshape_y, axis=1)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_pred[:, 1:], 6, 1)
        end_of_stroke = 1 / (1 + tf.math.exp(y_pred[:, 0]))
        pi_exp = tf.math.exp(pi_hat * (1 + self.b))
        pi_exp_sum = tf.reduce_sum(pi_exp, 1)
        pi = pi_exp / expand(pi_exp_sum, 1, self.M)
        sigma1 = tf.math.exp(sigma1_hat - self.b)
        sigma2 = tf.math.exp(sigma2_hat - self.b)
        rho = tf.math.tanh(rho_hat)

        gaussian = pi * bivariate_gaussian(expand(y1, 1, self.M), expand(y2, 1, self.M), mu1, mu2, sigma1, sigma2, rho)
        eps = 1e-20
        loss_gaussian = tf.reduce_sum(-tf.math.log(tf.reduce_sum(gaussian, 1) + eps))
        loss_bernoulli = tf.reduce_sum(
            -tf.math.log((end_of_stroke + eps) * y_end_of_stroke + (1 - end_of_stroke + eps) * (1 - y_end_of_stroke)))

        #   loss = (loss_gaussian + loss_bernoulli) / (tf.cast(batch_size, tf.float32) * args.T)
        loss = (loss_gaussian + loss_bernoulli) / self.T
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}




model = Model(rnn_state_size=64, numlayers=2, M=20, b=3.0, T=300, learning_rate=0.001)
model.compile()

x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)


print(tf.__version__)
