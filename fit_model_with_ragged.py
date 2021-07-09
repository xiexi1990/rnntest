import pickle
import keras
import tensorflow as tf
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

with open("x_y_100", "rb") as f:
    x, y = pickle.load(f)
    f.close()

xrag = tf.ragged.constant(x)
yarr = np.array(y)

stacked_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units=64) for _ in range(2)])
rnn_layer = tf.keras.layers.RNN(stacked_cell, return_state=False, return_sequences=False)

model = keras.Sequential([
    keras.layers.Input(shape=[None, 6], batch_size=1, dtype=tf.float32, ragged=True),
    rnn_layer,
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(xrag, yarr, epochs=100)


#print(model(x[0]))