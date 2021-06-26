import tensorflow as tf
import numpy as np

def expand(x, dim, N):
    _expand_dims_list = [tf.expand_dims(x, dim) for _ in range(N)]
    _concat_expand_dims_list = tf.concat(_expand_dims_list, dim)
    return _concat_expand_dims_list

def expand2(x, dim, N):
    _expand = tf.expand_dims(x, dim)
    _one_hot = tf.squeeze(tf.one_hot(indices=[dim], depth=tf.rank(_expand), on_value=N, off_value=1))
    return tf.tile(_expand, _one_hot)



arr1 = np.array([i for i in range(7)])
expand_arr1 = expand(arr1, 0, 3)
expand_expand_arr1 = expand(expand_arr1, 1, 2)

b = tf.placeholder(dtype = tf.int32, shape=None)

expand_arr2 = expand2(arr1, 0, b)
expand_expand_arr2 = expand2(expand_arr2, 1, 2)

# __expand1 = tf.expand_dims(arr1, 0)
# __tile1 = tf.tile(__expand1, [3, 1])
# __expand2 = tf.expand_dims(__tile1, 1)
#
# # d = tf.shape(__expand2)
# # for i in range(tf.rank(d)):
# #     d[i] = 1
# # d[1] = b
# d = tf.ones_like(tf.shape(__expand2), dtype=tf.int32)
# _one_hot = tf.squeeze(tf.one_hot(indices=[1], depth=tf.rank(__expand2), on_value=3))
# d += _one_hot
# __tile2 = tf.tile(__expand2, d)



with tf.Session() as sess:
    res1 = np.array(sess.run(expand_expand_arr1))
    print(res1)
    print('---------------------------------------')
   # res_shape = np.array(sess.run(_shape, feed_dict = {b:[3]}))
  #  print(res_shape)
    res2 = np.array(sess.run(expand_expand_arr2, feed_dict = {b:3}))

    print(res2)