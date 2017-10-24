# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#
def classification_rate(Y, P):
    prediction = predict(Y)
    return np.mean(Y == P)

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


def y2indicator2(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 2))
    for val in y:
        ind[val, ] = 1
    return ind
    
    



def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 2))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


#encoding the y-label
def M_B_0_1(label):
    if label == 'M':
        return 1
    else:
        return 0


# Criando setup

cancer_data = pd.read_csv('data.csv')
cancer_data['diagnosis'] = cancer_data['diagnosis'].apply(M_B_0_1)


cols_normalizar = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

cancer_data[cols_normalizar] = cancer_data[cols_normalizar].apply(lambda x: (x - np.mean(x)) / (np.std(x)  ) )

labels_to_drop = ['id','diagnosis']
x_data = cancer_data.drop(labels=labels_to_drop,axis=1)
y_data = cancer_data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)

#   
Y = cancer_data.iloc[:, 1].values
Y = np.matrix(cancer_data['diagnosis'])




# Encoding categorical data
#Maligo = 1  | Benigno = 0
for i in range(Y.shape[1]):
    if Y[0,i] == 'M':
        Y[0,i] = 1
    else:
        Y[0,i] = 0
Y = np.float64(Y)        

#################separando com sklearn train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=52)

Ytrain_ind = y2indicator2(y_train)
Ytest_ind = y2indicator(y_test)


# training setup 
max_iter = 20 # make it 30 for sigmoid
print_period = 10

lr = 0.00004
reg = 0.01

N, D = X_train.shape
batch_sz = 40  
n_batches = N // batch_sz

M = 100
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
        Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
        Ybatch = y_train_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
        # print "first batch cost:", cost(pYbatch, Ybatch)

        # updates
        W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
        b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
        W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
        b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

        if j % print_period == 0:
            # calculate just for LL
            pY, _ = forward(X_test, W1, b1, W2, b2)
            print "pY:", pY
            ll = cost(pY, Ytest_ind)
            LL_batch.append(ll)
            print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

            err = error_rate(pY, y_test)
            CR_batch.append(err)
            print "Error rate:", err

pY, _ = forward(X_test, W1, b1, W2, b2)
print "Final error rate:", error_rate(pY, y_test)
print " accuraccy: ", classification_rate(y_test,predict(pY))



##----------------------------------------------------------------------
#                         USANDO NUMPY


cancer_data = pd.read_csv('data.csv')

X = cancer_data.iloc[:, 2:-1].values

# 
Y = cancer_data.iloc[:, 1].values
Y = np.matrix(cancer_data.iloc[:, 1].values)

Y = cancer_data.iloc[:, 1].values
Y = np.matrix(cancer_data['diagnosis'])




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

Xtrain = X[:-150,]
Ytrain = Y[:-150]
Xtest  = X[-100:,]
Ytest  = Y[-100:]
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

N, D = Xtrain.shape
batch_sz = 40  
n_batches = N // batch_sz

M = 100
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
print " accuraccy: ", classification_rate(Ytest,predict(pY))


    

