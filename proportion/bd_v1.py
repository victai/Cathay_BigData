import numpy as np
import pandas as pd

def read_data():
	df = pd.read_csv('01.2011_Census_Microdata.csv', encoding='utf-8', dtype=str)
	return df
def read_input():
	fp = open('input_v1.txt', 'r')
	a = fp.readline()
	b = fp.readline()
	return a, b

def analyze(a, b, df):
	#print(len(a))
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

	cnt_b = b.count(';')+1
	b = b.strip().split(';')
	df3 = df
	for i in range(cnt_b):
		if len(b[i]) == 0: break
		feature_b = b[i].strip().split(' ')[0]
		condition_b = b[i].strip().split(' ', 1)[1].split(' ')
		x = (df.loc[:, feature_b] == condition_b[0])
		for j in range(1,len(condition_b)):
			x |= (df.loc[:, feature_b] == condition_b[j])
		df3 = df3.loc[x, : ]
	
	x = df2.shape[0]
	y = df3.shape[0]
	print(x, y)
	print("%.2f" % float(100*y/x) + '%')
	
if __name__ == '__main__':
	df = read_data()
	a, b = read_input()
	print('Big:  ', a)
	print('Small:', b)
	analyze(a, b, df)
