import pandas as pd
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
with open("label_cathay (1).csv") as f:
    data = csv.reader(f)
    y_train = list(data)

model = Sequential()
input_dense = Dense(units = 128, input_dim = len(x_train[0]), activation = 'relu')
model.add(input_dense)
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = SGD(lr = 0.1), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 1)
result = model.predict(x_train, batch_size = 32)
weight = input_dense.get_weights()
print(len(weight))
print(len(weight[0]))
print(len(weight[0][0]))
"""
with open("weight_cathay.csv", 'w', newline='') as csvfile:
	for a in range(len(weight[0])):
		for b in range(len(weight[0][0])):
 csvfile.write(weight)

"""

df = pd.DataFrame(weight[1])
df.to_csv('output1.csv', sep = '\t')
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#	print(df)
