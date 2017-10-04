# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


#

def predict(p_y):
    return np.argmax(p_y, axis=1)

def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

# MLP

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-( X.dot(W1) + b1 )))

    # rectifier
    # Z = X.dot(W1) + b1
    # Z[Z < 0] = 0
    # print "Z:", Z

    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    # print "Y:", Y, "are any 0?", np.any(Y == 0), "are any nan?", np.any(np.isnan(Y))
    # exit()
    return Y, Z

def derivative_w2(Z, T, Y):
    return Z.T.dot(Y - T)

def derivative_b2(T, Y):
    return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid
    #return X.T.dot( ( ( Y-T ).dot(W2.T) * (Z > 0) ) ) # for relu

def derivative_b1(Z, T, Y, W2):
    return (( Y-T ).dot(W2.T) * ( Z*(1 - Z) )).sum(axis=0) # for sigmoid
    #return (( Y-T ).dot(W2.T) * (Z > 0)).sum(axis=0) # for relu
############


def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 2))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


# Criando setup

dataframe = pd.read_csv('/home/vinicius/Repos/TFG/data.csv')
X = dataframe.iloc[:, 2:-1].values

# 
Y = dataframe.iloc[:, 1].values
Y = np.matrix(dataframe.iloc[:, 1].values)


# Encoding categorical data
#Maligo = 1  | Benigno = 0
for i in range(Y.shape[1]):
    if Y[0,i] == 'M':
        Y[0,i] = 1
    else:
        Y[0,i] = 0
Y = np.float64(Y)        


# normalization
for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

    

Y = Y.reshape(Y.shape[1],Y.shape[0])

    
max_iter = 20 # make it 30 for sigmoid
print_period = 10

lr = 0.00004
reg = 0.01

Xtrain = X[:-100,]
Ytrain = Y[:-100]
Xtest  = X[-100:,]
Ytest  = Y[-100:]
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

N, D = Xtrain.shape
batch_sz = 40  
n_batches = N // batch_sz

M = 40
K = 2
W1 = np.random.randn(D, M) / 28
b1 = np.zeros(M)
W2 = np.random.randn(M, K) / np.sqrt(M)
b2 = np.zeros(K)

# 1. batch
# cost = -16
LL_batch = []
CR_batch = []
for i in xrange(max_iter):
    for j in xrange(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
        Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
        # print "first batch cost:", cost(pYbatch, Ybatch)

        # updates
        W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
        b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
        W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
        b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

        if j % print_period == 0:
            # calculate just for LL
            pY, _ = forward(Xtest, W1, b1, W2, b2)
            print "pY:", pY
            ll = cost(pY, Ytest_ind)
            LL_batch.append(ll)
            print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

            err = error_rate(pY, Ytest)
            CR_batch.append(err)
            print "Error rate:", err

pY, _ = forward(Xtest, W1, b1, W2, b2)
print "Final error rate:", error_rate(pY, Ytest)


    

