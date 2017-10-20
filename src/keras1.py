import numpy as np
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

with open("train_cathay (1).csv") as f:
    data = csv.reader(f)
    x_train = list(data)
with open("label_cathay (1).csv") as f:
    data = csv.reader(f)
    y_train = list(data)

model = Sequential()
model.add(Dense(units = 1024, input_dim = len(x_train[0]), activation = 'relu'))
model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = SGD(lr = 0.1), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 1)
result = model.predict(x_train, batch_size = 32)

