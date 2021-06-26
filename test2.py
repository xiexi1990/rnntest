import tensorflow as tf
import tensorflow.compat.v1 as v1

g = v1.Graph()


with g.as_default():
  in_a = v1.placeholder(dtype=v1.float32, shape=(2,1))
  in_b = v1.placeholder(dtype=v1.float32, shape=(2,1))

  def forward(x):
    with v1.variable_scope("matmul", reuse=v1.AUTO_REUSE):
      W = v1.get_variable("W", initializer=v1.ones(shape=(2,2)),
                          regularizer=lambda x:tf.reduce_mean(x**2))
      b = v1.get_variable("b", initializer=v1.zeros(shape=(2,1)))
      return tf.matmul(W , x) + b

  out_a = forward(in_a)
  out_b = forward(in_b)
  reg_loss=v1.losses.get_regularization_loss(scope="matmul")

config = v1.ConfigProto()
config.gpu_options.allow_growth = True

with v1.Session(graph=g, config=config) as sess:
  sess.run(v1.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss],
                feed_dict={in_a: [[1], [0]], in_b: [[0], [1]]})

print(outs[0])
print()
print(outs[1])
print()
print(outs[2])