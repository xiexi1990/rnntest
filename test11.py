import numpy as np
import random
import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

x = []
#y = []
y = np.zeros([30, 5])
for i in range(0, 30):
    x_sample_len = random.randint(3, 7)
    x_sample = np.random.random([x_sample_len, 6])
    x.append(x_sample)
    y_class = random.randint(0,4)
    y[i, y_class] = 1
  #  y.append(y_class)



xrag = tf.ragged.constant(x)
max_seq = xrag.bounding_shape()[-1]


model = keras.Sequential([
    keras.layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True),

    keras.layers.LSTM(32),
    keras.layers.Dense(5, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=True))
history = model.fit(xrag, y, epochs=5)
print(history)