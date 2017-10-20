import numpy as np
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras import layers
#np.set_printoptions(threshold=np.inf)
with open("train_cathay (1).csv") as f:
    data = csv.reader(f)
    x_train = list(data)
with open("multiLabel_cathay.csv") as f:
    data = csv.reader(f)
    y_train = list(data)

model = Sequential()
input_dense = Dense(units = 1028, input_dim = len(x_train[0]), activation = 'relu')
model.add(input_dense)
model.add(Dense(units = 1028, input_dim = len(x_train[0]), activation = 'relu'))
model.add(Dense(units = 6, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr = 0.1), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 64, epochs = 16)
result = model.predict(x_train, batch_size = 64)
weight = input_dense.get_weights()
print(len(weight))
print(len(weight[0]))
print(len(weight[0][0]))


