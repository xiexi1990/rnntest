import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import random
from typing import Union
from math import ceil
from os import mkdir
import pickle

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


class S_LSTM(keras.layers.Layer):
    def __init__(self, y_class, h_size, nlayer, **kwargs):
        super().__init__(**kwargs)
        self.y_class = y_class
        self.h_size = h_size
        self.nlayer = nlayer
        self.Wxi, self.Wxf, self.Wxc, self.Wxo = [], [], [], []
        self.Whi, self.Whf, self.Whc, self.Who = [], [], [], []
        self.Whsi, self.Whsf, self.Whsc, self.Whso = [], [], [], []
        self.Wc = []
        self.bi, self.bf, self.bc, self.bo = [], [], [], []

    def build(self, input_shape):
        for i in range(self.nlayer):
            self.Wxi.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer="random_normal"))
            self.Wxf.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer="random_normal"))
            self.Wxc.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer="random_normal"))
            self.Wxo.append(self.add_weight(shape=(input_shape[2], self.h_size), initializer="random_normal"))
            if i > 0:
                self.Whsi.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
                self.Whsf.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
                self.Whsc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
                self.Whso.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.Whi.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.Whf.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.Whc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.Who.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.Wc.append(self.add_weight(shape=(self.h_size, self.h_size), initializer="random_normal"))
            self.bi.append(self.add_weight(shape=(1, self.h_size), initializer="random_normal"))
            self.bf.append(self.add_weight(shape=(1, self.h_size), initializer="random_normal"))
            self.bc.append(self.add_weight(shape=(1, self.h_size), initializer="random_normal"))
            self.bo.append(self.add_weight(shape=(1, self.h_size), initializer="random_normal"))
         #   self.h.append(tf.Variable(tf.zeros([1, self.h_size]), shape=[None, self.h_size]))
         #   self.c.append(tf.Variable(tf.zeros([1, self.h_size]), shape=[None, self.h_size]))
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs):
        def time_step(prev, xt):
            c_1, h_1 = tf.unstack(prev)
            c, h = [], []
            for i in range(self.nlayer):
                _i = tf.math.sigmoid(xt * self.Wxi[i] + (i > 0 and h[i - 1] * self.Whsi[i - 1]) + h_1[i] * self.Whi[i] + self.bi[i])
                _f = tf.math.sigmoid(xt * self.Wxf[i] + (i > 0 and h[i - 1] * self.Whsf[i - 1]) + h_1[i] * self.Whf[i] + self.bf[i])
                _o = tf.math.sigmoid(xt * self.Wxo[i] + (i > 0 and h[i - 1] * self.Whso[i - 1]) + h_1[i] * self.Who[i] + self.bo[i])
                _c =    tf.math.tanh(xt * self.Wxc[i] + (i > 0 and h[i - 1] * self.Whsc[i - 1]) + h_1[i] * self.Whc[i] + self.bc[i])
                c.append(tf.math.multiply(_f, c_1[i]) + tf.math.multiply(_i, _c))
                h.append(tf.math.multiply(_o, tf.math.tanh(c[i])))
            return tf.stack([c, h])

        outputs = tf.scan(time_step, inputs)
        return outputs



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

    def save(self) -> None:
        with open("simple-model-weights", 'wb') as f:
            pickle.dump((self.vocab, self.a_size, self.wa, self.ba, self.wy, self.by), f, protocol=4)
            f.close()


    def load(self) -> None:
        with open("simple-model-weights", 'rb') as f:
            self.vocab, self.a_size, self.wa, self.ba, self.wy, self.by = pickle.load(f)
            f.close()
            self.vocab_size = len(self.vocab)
            self.combined_size = self.vocab_size + self.a_size
            self.weights = [self.wa, self.ba, self.wy, self.by]

        # self.wa = tf.Variable(np.load(f'./{name}/wa.npy'))
        # self.ba = tf.Variable(np.load(f'./{name}/ba.npy'))
        # self.wy = tf.Variable(np.load(f'./{name}/wy.npy'))
        # self.by = tf.Variable(np.load(f'./{name}/by.npy'))
        # self.weights = [self.wa, self.ba, self.wy, self.by]



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
#model.fit(sentences, batch_size=128, epochs=2)

#model.save()
model.load()

for i in range(20):
    print(model.sample())

