#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 03:11:43 2021

@author: shaan
"""

# import libraries
import numpy as np			
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from decimal import *

# generate data (inputs x and outputs f(x)) 
def generate_data(means, sigma, ndatapoints):
    nclasses = 2
    data = np.zeros( (nclasses * ndatapoints, 3) )
    for c in range(0, nclasses):
        starti = c * ndatapoints
        endi = (c + 1) * ndatapoints
        data[starti:endi, 0:2] = means[c] + sigma * random.standard_normal( (ndatapoints, 2) )
        data[starti:endi, 2] = c
    randvec = np.random.permutation(nclasses * ndatapoints)    
    data = data[randvec,:]
    return data, randvec;

# set parameters and generate training data   
means = (0.3, 0.7)
sigma = 0.09
ndatapoints = 128 # generating 128 training examples
data_output_train = generate_data(means, sigma, ndatapoints)  
data_train = data_output_train[0]
randvec_train = data_output_train[1]

# show generated data
colors_train = np.concatenate( (np.matlib.repmat(np.array([1, 0.5, 1]), ndatapoints, 1), np.matlib.repmat(np.array([0.5, 0.5, 1]), ndatapoints, 1)))
colors_train = colors_train[randvec_train,:]
figi_train = 1 
plt.figure(figi_train)
plt.clf()
plt.scatter(data_train[:,0], data_train[:,1], c = colors_train, alpha = 0.5)
plt.axis('square')  
plt.xlabel('x1 (0 = green, 1 = red)')
plt.ylabel('x2 (0 = small, 1 = large)')
plt.title('classes of apples (training data)')

# sigmoid activation function
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))    
    return a

# show the sigmoid function 
z = np.linspace(-8, 8, 100)
a = sigmoid(z)
plt.figure(2)
plt.clf()
plt.plot(z, a)
plt.xlabel('z (input)')
plt.ylabel('a (output)')
plt.title('sigmoid activation function')

# initialize weights
def initialize_weights(nweights, randn):
    if randn == 0:
        w = np.zeros( (nweights, 1) )       
    else:
        w = 0.001 * random.standard_normal( (nweights, 1) )
    b = 0
    return w, b

nweights = 2
randn = 0    
w, b = initialize_weights(nweights, randn)

# propagate forward    
def forward(X, w, b): 
    z = np.dot(np.transpose(w), X) + b
    a = sigmoid(z)
    return a

X = np.transpose( data_train[:,0:2] )
a = forward(X, w, b)

# compute cost
def compute_cost(a, y):
    m = a.shape[1]
    cost =  -(1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    return cost
 
y = np.transpose( data_train[:,2:3] )  
cost = compute_cost(a, y) 
 
# show cost function
a_ls = np.linspace(0.0001, 0.9999, 100)
cost_y1 = - np.log(a_ls)
cost_y0 = - np.log(1 - a_ls)
plt.figure(3)
plt.clf()
plt.plot(a_ls, cost_y1)
plt.plot(a_ls, cost_y0)
plt.legend( ('if y = 1: -log(a)', 'if y = 0: -log(1-a)') )
plt.xlabel('a = f(x)')
plt.ylabel('cost')
plt.title('cost function')

# show error surface
def error_surface(b, w, cost, figi):
    # plot error surface across w1 and w2 for a user-identified value of b (input arg b)
    w1 = np.linspace(-15, 15, 100)
    w2 = np.linspace(-15, 15, 100)
    xx, yy = np.meshgrid(w1, w2)
    costs = np.zeros( (w1.shape[0], w2.shape[0]) )
    for r in range(w1.shape[0]):
        for c in range(w2.shape[0]):
            cw = np.zeros( (nweights, 1) )
            cw[0] = xx[r,c]
            cw[1] = yy[r,c]
            ca = forward(X, cw, b)
            ccost = compute_cost(ca, y)
            costs[r,c] = ccost
    plt.figure(figi)
    plt.clf()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(xx, yy, costs, cmap = 'binary', edgecolor = 'none')
    ax.scatter(w[0], w[1], cost, c = 'r') # plot cost for current weights (input args b and w)
	
    # add figure labels
    width = 3
    precision = 3
    value = Decimal(b)
    title = f"error surface for b = {value:{width}.{precision}}"
    ax.set_title(title)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('cost')
    plt.show()
    plt.pause(0.004)
    
    return xx, yy, costs 

xx, yy, costs = error_surface(b, w, cost, 4)

# propagate back  
def back(a, y, X):  
    m = a.shape[1]     
    dw = (1 / m) * np.dot(X, np.transpose(a - y))
    db = (1 / m) * np.sum(a - y)
    return dw, db
    
dw, db = back(a, y, X)

# plot the decision boundary (as in lab 9)
def plot_boundary(weights, figi):
    b = weights['b']
    w = weights['w']    
    slope = -(b / w[1]) / (b / w[0])
    y_intercept = -b / w[1]
    x = np.linspace(0,1,100)
    y = (slope * x) + y_intercept
    plt.figure(figi)
    plt.plot(x, y)
    plt.pause(0.004)
    
# optimize weights using gradient descent
def optimize(w, b, X, y, niterations, alpha, monitor):
  
    costs = []
    
    for i in range(niterations):
        a = forward(X, w, b)
        cost = compute_cost(a, y)
        dw, db = back(a, y, X)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        costs.append(cost)

	# plot decision boundary (for first 100 iterations and last iteration)
        if monitor and (i <= 100 or i == niterations-1):
            cweights = {"w": w,
                       "b": b}
            plot_boundary(cweights, 1)        

	# print cost and show error surface (incl. cost) for every 100 iterations
        if monitor and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            error_surface(b, w, cost, 4)            
                
    weights = {"w": w,
              "b": b}
              
    gradients = {"dw": dw,
                "db": db}
    
    return weights, gradients, costs           

niterations = 2000
alpha = 0.5
monitor = 1
weights, gradients, costs = optimize(w, b, X, y, niterations, alpha, monitor)

# plot cost as a function of iteration
plt.figure(5)
plt.clf()
plt.plot(costs)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('minimizing cost through gradient descent')

# generate test data   
means = (0.3,0.7)
sigma = 0.09
ndatapoints = 128
data_output_test = generate_data(means, sigma, ndatapoints)  
data_test = data_output_test[0]
randvec_test = data_output_test[1]

# show generated data and learned decision boundary
colors_test = np.concatenate( (np.matlib.repmat(np.array([1, 0.5, 1]), ndatapoints, 1), np.matlib.repmat(np.array([0.5, 0.5, 1]), ndatapoints, 1)))
colors_test = colors_test[randvec_test,:]
figi_test = 6; 
plt.figure(figi_test)
plt.clf()
plt.scatter(data_test[:,0], data_test[:,1], c = colors_test, alpha = 0.5)
plt.axis('square')  
plt.xlabel('x1 (0 = green, 1 = red)')
plt.ylabel('x2 (0 = small, 1 = large)')
plt.title('classes of apples (test data)')
plot_boundary(weights, figi_test)

# compute class predictions
def predict(data, weights):
    X = np.transpose( data[:,0:2] )
    y = np.transpose(data[:,2:3]) 

    a = forward(X, weights['w'], weights['b'])
    p = np.zeros(a.shape)
    for i in range(a.shape[1]):
        if a[0,i] < 0.5:
            p[0,i] = 0
        else:
            p[0,i] = 1

    return p

y_test = np.transpose(data_test[:,2:3])
p_test = predict(data_test, weights)
print("test accuracy: {} %".format(100 - np.mean(np.abs(p_test - y_test)) * 100))

