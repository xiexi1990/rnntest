from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np

model = Sequential()

model.add(LSTM(6, return_sequences=True, input_shape=(None, 1)))
model.add(LSTM(6, return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='sigmoid')))

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

def train_generator():
    while True:
        sequence_length = np.random.randint(8, 16)
        x_train = np.random.random((10, sequence_length, 1))
        # y_train will depend on past 5 timesteps of x

        y_train = to_categorical(x_train > 0.5)
        yield x_train, y_train

model.fit_generator(train_generator(), steps_per_epoch=10, epochs=5, verbose=1)