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

model = keras.Sequential([
    keras.layers.Input(shape=[None, 6], batch_size=1, dtype=tf.float32, ragged=True),

    keras.layers.LSTM(64),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(xrag, yarr, epochs=1000)


#print(model(x[0]))