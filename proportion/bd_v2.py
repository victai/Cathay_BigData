import numpy as np
import pandas as pd

def read_data():
	df = pd.read_csv('01.2011_Census_Microdata.csv', encoding='utf-8', dtype=str)
	return df
def read_input():
	fp = open('input_v2.txt', 'r')
	a = fp.readline()
	return a

def analyze(a, df):
	fp = open('input_v2.txt', 'r')
	for a in fp:
		print(a)
		cnt_a = a.count(';')+1
		a = a.strip().split(';')
		df2 = df
		for i in range(cnt_a):
			if len(a[i]) == 0:	break
			feature_a = a[i].strip().split(' ')[0]
			condition_a = a[i].strip().split(' ', 1)[1].split(' ')
			x = (df.loc[:, feature_a] == condition_a[0])
			for j in range(1,len(condition_a)):
				x |= (df.loc[:, feature_a] == condition_a[j])
			df2 = df2.loc[x, : ]
		arr = np.zeros(6)
		for i in range(5):
			x = (df2.loc[:, 'Health'] == str(i+1))
			tmp = df2.loc[x, :]
			arr[i] = tmp.shape[0]

		y = df2.shape[0]
		tmp2 = y
		for i in range(5):
			tmp2 -= arr[i]
			print("%.2f" % float(100*arr[i]/y) + '%')
		print("%.2f" % float(100*tmp2/y) + '%')
		print('-'*50)
	
if __name__ == '__main__':
	df = read_data()
	a = read_input()
	print(a)
	analyze(a, df)
