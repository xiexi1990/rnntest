import tensorflow as tf
import numpy as np
import os
import time
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
text = text[:len(text)//20]
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
seq_length = 100
examples_per_epoch = len(text)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(1):
  print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
# 批大小
BATCH_SIZE = 64
# 设定缓冲区大小，以重新排列数据集
# （TF 数据被设计为可以处理可能是无限的序列，
# 所以它不会试图在内存中重新排列整个序列。相反，
# 它维持一个缓冲区，在缓冲区重新排列元素。）
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)
embedding_dim = 64
rnn_units = 64
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

checkpoint_dir = './training_checkpoints'
if False:
    model = build_model(
      vocab_size = len(vocab),
      embedding_dim=embedding_dim,
      rnn_units=rnn_units,
      batch_size=BATCH_SIZE)
    def loss(labels, logits):
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # 检查点保存至的目录

    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    EPOCHS=10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  num_generate = 1000
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  # 低温度会生成更可预测的文本
  # 较高温度会生成更令人惊讶的文本
  temperature = 1.0
  # 这里批大小为 1
  hid = model.reset_states()

  print(model(np.array([[2]])))
  print(model(np.array([[2]])))
  print(model(np.array([[2]])))



  for i in range(num_generate):
      predictions = model(input_eval)
      # 删除批次的维度
      predictions = tf.squeeze(predictions, 0)
      # 用分类分布预测模型返回的字符
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))