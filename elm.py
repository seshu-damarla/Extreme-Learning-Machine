# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:22:48 2022

@author: Seshu Kumar Damarla
"""
"""
Python program for Extreme Learning Machine
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# data 
xtrain = pd.read_csv('trainxdata.csv',header=None)
ytrain=pd.read_csv('trainydata.csv',header=None)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtest = pd.read_csv('testxdata.csv', header=None)
ytest = pd.read_csv('testydata.csv', header=None)

xtest = np.array(xtest)
ytest = np.array(ytest)

# normlization (z-score)
xmean = np.mean(xtrain, axis=0, keepdims=True,dtype=np.float)
xstd = np.std(xtrain, axis=0, keepdims=True,dtype=np.float) 
ymean = np.mean(ytrain, axis=0, keepdims=True,dtype=np.float)
ystd = np.std(ytrain, axis=0, keepdims=True,dtype=np.float)

xtrain = (xtrain-xmean)/xstd
ytrain = (ytrain-ymean)/ystd

#print(xmean)
#print(xstd)
#print(ymean)
#print(ystd)

# weights and biases of neurons in the hidden layer
nh=35                # no. of neurons in the hidden layer
ni=xtrain.shape[1]   # no. of features 
nex=400              # no. of training examples 
W=np.random.randn(nh,ni)  # weights of neurons in the hidden layer
B=np.random.randn(nh,1)   # biase of neurons in the hidden layer

# output of hidden layer
h = np.dot(W,xtrain.T) + B
H=np.exp(-h**2)

# weights of output layer
Hinv = np.linalg.pinv(H)
beta=np.dot(ytrain.T,Hinv)

# predictions on test data
xtest = (xtest-xmean)/xstd
htest = np.dot(W,xtest.T) + B
Htest = np.exp(-htest**2)

ypred = np.dot(beta,Htest)
ypred = ypred.T
ypred = ypred*ystd+ymean

plt.plot(ypred)
plt.plot(ytest)
(R, pval) = stats.pearsonr(ytest.flatten(),ypred.flatten())
print(R)







