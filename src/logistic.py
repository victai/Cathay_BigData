import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=np.inf)

sigmoid = lambda z : 1/(1+ np.exp(-z))

def parse_output(w):
	f = open('../proportion/input_v2.txt', 'r')
	c = f.read().split('\n')
	for i in range(6):
		del c[42]
	del c[101]
	a = np.arange(0,101).reshape(101, 1)
	a = np.hstack((a,w))
	a = a[a[:,1].argsort()]
	for i in range(101):
		print('%-30s    %10f' % (c[int(a[i][0])], a[i][1]))

def Read_train():
	X = pd.read_csv('../data/train_cathay_all.csv', encoding='Big5').values[:,1:]
	Y = pd.read_csv('../data/label_cathay_all.csv', encoding='Big5').values
	#X = np.delete(X, (42,43,44,45,46,47), axis=1)
	#pd.DataFrame(X).to_csv('new_train_cathay_1.csv')
	X_1 = X[200000:, :]
	X_2 = X[:200000, :]
	Y_1 = Y[200000:, :]
	Y_2 = Y[:200000, :]

	return X_1, X_2, Y_1, Y_2

def Test(w, b, X, Y):
	f = sigmoid(X_2.dot(w)+b)
	pred = np.zeros([X_2.shape[0], 1], dtype=int)
	pred[f > 0.5] = 1
	#print(np.count_nonzero(pred==Y)/Y.shape[0])

def Logistic_regression(X, Y):
	w = np.zeros([101, 1])
	b = 0
	sum_w_grad = np.zeros([101, 1])
	sum_b_grad = 0
	iteration = 200 
	lr = 0.1
	for i in range(iteration):
		f = sigmoid(X.dot(w) + b)
		w_grad = -X.T.dot(Y-f)
		b_grad = -np.sum(Y-f)
		sum_w_grad += np.square(w_grad)
		sum_b_grad += np.square(b_grad)
		w -= lr*w_grad/np.sqrt(sum_w_grad)
		b -= b_grad/np.sqrt(sum_b_grad)
		pred = np.zeros([X.shape[0], 1], dtype=int)
		pred[f > 0.5] = 1	
		#print('\r', i, np.count_nonzero(pred==Y)/Y.shape[0], end='\n')
		a = np.count_nonzero(pred==Y)
		aa = Y.shape[0]
	#print()
	return w, b

if __name__ == '__main__':
	X_1, X_2, Y_1, Y_2= Read_train()
	w, b = Logistic_regression(X_1, Y_1)
	Test(w, b, X_2, Y_2)
	np.around(w, decimals=4)
	parse_output(w)
