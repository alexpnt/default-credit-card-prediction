#!/usr/bin/env python2
# -*- coding: utf-8 -*

import numpy as np

def load_dataset(filename):
	#open file
	f=open(filename)

	#skip headers
	features=f.readline().rstrip("\n").split(",")
	feature_names=f.readline().rstrip("\n").split(",")

	dataset = np.loadtxt(f,delimiter = ',')
	dataset=dataset.astype(int)
	f.close()

	#get trainning /(X) and target (y) data
	X=dataset[:,0:23]
	y=dataset[:,23]

	return features,feature_names,X,y

if __name__ == '__main__':
	load_dataset('dataset/default of credit card clients.csv')

