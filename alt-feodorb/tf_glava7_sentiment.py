# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:02:21 2019

@author: feodorb
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

top_words=5000
max_review_length = 500

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# restore np.load for future normal usage
np.load = np_load_old

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length = max_review_length))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose= 0)
print('Accuracy: %.4f' % (scores[1]))


