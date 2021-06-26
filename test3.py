import tensorflow as tf
import tensorflow.compat.v1 as v1

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def model(x, training, scope='model'):
  with v1.variable_scope(scope, reuse=v1.AUTO_REUSE):
    x = v1.layers.conv2d(x, 32, 3, activation=v1.nn.relu,
          kernel_regularizer=lambda x:0.004*tf.reduce_mean(x**2))
    x = v1.layers.max_pooling2d(x, (2, 2), 1)
    x = v1.layers.flatten(x)
    x = v1.layers.dropout(x, 0.1, training = False)
    x = v1.layers.dense(x, 64, activation=v1.nn.relu)
    x = v1.layers.batch_normalization(x, training = True)
    x = v1.layers.dense(x, 10)
    return x

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)

print(train_out)
print()
print(test_out)