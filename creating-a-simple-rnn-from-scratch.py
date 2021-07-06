import numpy as np
import pandas as pd
import tensorflow as tf
import random
from typing import Union
from math import ceil
from os import mkdir

UNK = '<UNK>' # Unknown word
EOS = '<EOS>' # End of sentence

def build_vocabulary(sentences: list, words_to_keep: int) -> list:
    # builds a vocabulary using 'words_to_keep' most frequent words
    # encountered in the list of sentences
    vocabulary = {}
    n = len(sentences)
    for i, s in enumerate(sentences):
        print('Creating vocabulary: %05.2f%%' % (100*(i+1)/n,), end='\r')
        for word in s.strip().split():
            vocabulary[word] = vocabulary.get(word, 0) + 1
    vocabulary = list(vocabulary.items())
    vocabulary.sort(reverse=True, key=lambda e: e[1])
    vocabulary = vocabulary[0:words_to_keep]
    vocabulary = [e[0] for e in vocabulary]
    vocabulary.sort()
    vocabulary.append(UNK)
    vocabulary.append(EOS)
    print('Done'+(50*' '))
    return vocabulary

def build_sentences(vocabulary: list, sentences: list) -> list:
    # transforms the list of sentences into a list of lists of words
    # replacing words that are not in the vocabulary with <UNK>
    # and appending <EOS> at the end of each sentence
    processed_sent = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        print('Creating sentences list: %05.2f%%' % (100*(i+1)/n,), end='\r')
        s = []
        for word in sent.strip().split():
            if word not in vocabulary:
                word = UNK
            s.append(word)
        s.append(EOS)
        processed_sent.append(s)
    print('Done'+(50*' '))
    return processed_sent

def word2index(vocabulary: list, word: str) -> int:
    # returns the index of 'word' in the vocabulary
    return vocabulary.index(word)

def words2onehot(vocabulary: list, words: list) -> np.ndarray:
    # transforms the list of words given as argument into
    # a one-hot matrix representation using the index in the vocabulary
    n_words = len(words)
    n_voc = len(vocabulary)
    indices = np.array([word2index(vocabulary, word) for word in words])
    a = np.zeros((n_words, n_voc))
    a[np.arange(n_words), indices] = 1
    return a

def sample_word(vocabulary: list, prob: np.ndarray) -> str:
    # sample a word from the vocabulary according to 'prob'
    # probability distribution (the softmax output of our model)
    # until it is != <UNK>
    while True:
        word = np.random.choice(vocabulary, p=prob)
        if word != UNK:
            return word


class Model:
    def __init__(self, vocabulary: list = [], a_size: int = 0):
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)
        self.a_size = a_size
        self.combined_size = self.vocab_size + self.a_size

        # weights and bias used to compute the new a
        # (a = vector that is passes to the next time step)
        self.wa = tf.Variable(tf.random.normal(
            stddev=1.0 / (self.combined_size + self.a_size),
            shape=(self.combined_size, self.a_size),
            dtype=tf.double))
        self.ba = tf.Variable(tf.random.normal(
            stddev=1.0 / (1 + self.a_size),
            shape=(1, self.a_size),
            dtype=tf.double))

        # weights and bias used to compute y (the softmax predictions)
        self.wy = tf.Variable(tf.random.normal(
            stddev=1.0 / (self.a_size + self.vocab_size),
            shape=(self.a_size, self.vocab_size),
            dtype=tf.double))
        self.by = tf.Variable(tf.random.normal(
            stddev=1.0 / (1 + self.vocab_size),
            shape=(1, self.vocab_size),
            dtype=tf.double))

        self.weights = [self.wa, self.ba, self.wy, self.by]
        self.optimizer = tf.keras.optimizers.Adam()

    def __call__(self,
                 a: Union[np.ndarray, tf.Tensor],
                 x: Union[np.ndarray, tf.Tensor],
                 y: Union[np.ndarray, tf.Tensor, None] = None) -> tuple:

        a_new = tf.math.tanh(tf.linalg.matmul(tf.concat([a, x], axis=1), self.wa) + self.ba)
        y_logits = tf.linalg.matmul(a_new, self.wy) + self.by
        if y is None:
            # during prediction return softmax probabilities
            return (a_new, tf.nn.softmax(y_logits))
        else:
            # during training return loss
            return (a_new, tf.math.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y, y_logits)))

    def fit(self,
            sentences: list,
            batch_size: int = 128,
            epochs: int = 10) -> None:

        n_sent = len(sentences)
        num_batches = ceil(n_sent / batch_size)

        for epoch in range(epochs):

            random.shuffle(sentences)
            start = 0
            batch_idx = 0

            while start < n_sent:

                print('Training model: %05.2f%%' % (100 * (epoch * num_batches + batch_idx + 1) / (epochs * num_batches),))

                batch_idx += 1
                end = min(start + batch_size, n_sent)
                batch_sent = sentences[start:end]
                start = end
                batch_sent.sort(reverse=True, key=lambda s: len(s))

                init_num_words = len(batch_sent)
                a = np.zeros((init_num_words, self.a_size))
                x = np.zeros((init_num_words, self.vocab_size))

                time_steps = len(batch_sent[0])

                with tf.GradientTape() as tape:

                    losses = []
                    for t in range(time_steps):
                        words = []
                        for i in range(init_num_words):
                            if t >= len(batch_sent[i]):
                                break
                            words.append(batch_sent[i][t])

                        y = words2onehot(self.vocab, words)
                        n = y.shape[0]
                        a, loss = self(a[0:n], x[0:n], y)
                        losses.append(loss)
                        x = y

                    loss_value = tf.math.reduce_mean(losses)

                grads = tape.gradient(loss_value, self.weights)
                self.optimizer.apply_gradients(zip(grads, self.weights))

    def sample(self) -> str:
        # sample a new sentence from the learned model
        sentence = ''
        a = np.zeros((1, self.a_size))
        x = np.zeros((1, self.vocab_size))
        while True:
            a, y_hat = self(a, x)
            word = sample_word(self.vocab, tf.reshape(y_hat, (-1,)))
            if word == EOS:
                break
            sentence += ' ' + word
            x = words2onehot(self.vocab, [word])
        return sentence[1:]

    def predict_next(self, sentence: str) -> str:
        # predict the next part of the sentence given as parameter
        a = np.zeros((1, self.a_size))
        for word in sentence.strip().split():
            if word not in self.vocab:
                word = UNK
            x = words2onehot(self.vocab, [word])
            a, y_hat = self(a, x)
        s = ''
        while True:
            word = sample_word(self.vocab, tf.reshape(y_hat, (-1,)))
            if word == EOS:
                break
            s += ' ' + word
            x = words2onehot(self.vocab, [word])
            a, y_hat = self(a, x)
        return s

    def save(self, name: str) -> None:
        mkdir(f'./{name}')
        with open(f'./{name}/vocabulary.txt', 'w') as f:
            f.write(','.join(self.vocab))
        with open(f'./{name}/a_size.txt', 'w') as f:
            f.write(str(self.a_size))
        np.save(f'./{name}/wa.npy', self.wa.numpy())
        np.save(f'./{name}/ba.npy', self.ba.numpy())
        np.save(f'./{name}/wy.npy', self.wy.numpy())
        np.save(f'./{name}/by.npy', self.by.numpy())

    def load(self, name: str) -> None:
        with open(f'./{name}/vocabulary.txt', 'r') as f:
            self.vocab = f.read().split(',')
        with open(f'./{name}/a_size.txt', 'r') as f:
            self.a_size = int(f.read())

        self.vocab_size = len(self.vocab)
        self.combined_size = self.vocab_size + self.a_size

        self.wa = tf.Variable(np.load(f'./{name}/wa.npy'))
        self.ba = tf.Variable(np.load(f'./{name}/ba.npy'))
        self.wy = tf.Variable(np.load(f'./{name}/wy.npy'))
        self.by = tf.Variable(np.load(f'./{name}/by.npy'))
        self.weights = [self.wa, self.ba, self.wy, self.by]



df = pd.read_csv('abcnews-date-text.csv', nrows=5000)


vocabulary = build_vocabulary(
                 df['headline_text'].values.tolist(),
                 words_to_keep=500)
sentences = build_sentences(
                vocabulary,
                df['headline_text'].values.tolist())

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model = Model(vocabulary, 1024)
model.fit(sentences, batch_size=128, epochs=2)

model.save('news_headlines_model')
# model.load('news_headlines_model')

for i in range(20):
    print(model.sample())

