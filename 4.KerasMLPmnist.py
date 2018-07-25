# 4.2 How to initialize weights

from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = x_train.reshape([-1, 28*28]) / 255.
X_test = x_test.reshape([-1, 28*28]) / 255.

model = Sequential()
model.add(Dense(100, input_shape=(28*28,), activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(10, activation='softmax'))


# 1
model.compile(loss='categorical_crossentropy', init="he_uniform", optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1, validation_data=(X_test, Y_test))

# 2
model.compile(loss='categorical_crossentropy', init="glorot_normal", optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1, validation_data=(X_test, Y_test))
