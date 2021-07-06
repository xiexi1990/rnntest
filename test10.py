import tensorflow as tf
max_features = 20000
batch_size = 32
BUFFER_SIZE=1000

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=max_features,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3)

r_train_x = tf.ragged.constant(x_train)
r_test_x = tf.ragged.constant(x_test)

keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
    tf.keras.layers.Embedding(max_features,128),
    tf.keras.layers.LSTM(32, use_bias=False),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation(tf.nn.relu),
    tf.keras.layers.Dense(1)
])

NumEpochs = 10
BatchSize = 32

keras_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = keras_model.fit(r_train_x, y_train, epochs=NumEpochs, batch_size=BatchSize, validation_data=(r_test_x, y_test))