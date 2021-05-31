'''
AM4264 Problem Set 9
Shaan Verma
250804514
'''

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import numpy.matlib

# main function
def ps9(peri,monitor):
    # seed the random number generator (to ensure that we all generate the same data and initial weights)
    random.seed(3)

    # generate training data
    means = np.array([[0.3,0.3],[0.3,0.75],[0.75,0.3],[0.75,0.75]])
    sigma = 0.04
    ndatapoints = 20
    data_output_train = generate_data_4classes(means, sigma, ndatapoints)
    data_train = data_output_train[0]
    randvec_train = data_output_train[1]

    # training parameters
    learning_rate = 0.01
    niterations = 2

    # train perceptron 1 (OR)
    training_output = train(data_train[:,[0,1,2]], learning_rate, niterations)
    weights_OR = training_output[0]

    # train perceptron 2 (AND)    solution_code_ps9.py
    training_output = train(data_train[:,[0,1,3]],learning_rate, niterations)
    weights_AND = training_output[0]

    # train perceptron 3 (XOR)
    # this perceptron takes the outputs from perceptron 1 and 2 as input (= as training data)
    # assemble the training data (complete this yourself using the test function - needs 2-3 lines of code)
    predictions_OR = test(data_train[:,[0,1,2]],weights_OR)
    predictions_AND = test(data_train[:,[0,1,3]],weights_AND)

    # Combining predictions from OR and AND with training data for XOR
    holder = (np.vstack((predictions_OR,predictions_AND)).T)
    holder = np.hstack((holder,data_train[:,[4]]))

    training_output = train(holder,learning_rate,niterations)
    weights_XOR = training_output[0]

    # show training data and decision boundaries for the three perceptrons
    if monitor:
        # plt.ion() # you may need to turn interactive mode on for figure plotting
        # perceptron 1 (OR)
        colors = plot_data(ndatapoints, data_output_train, 1)
        plot_boundary(weights_OR, 1)
        # perceptron 2 (AND)
        colors = plot_data(ndatapoints, data_output_train, 2)
        plot_boundary(weights_AND, 2)
        # perceptron 3 (XOR)
        plot_data_XOR(colors, predictions_OR, predictions_AND, 3)
        plot_boundary(weights_XOR, 3)

    # print weights
    if peri == 1:
        print(weights_OR)
    if peri == 2:
        print(weights_AND)
    if peri == 3:
        print(weights_XOR)

# helper functions
# generate data for 4 classes (input x and output f(x))
def generate_data_4classes(means, sigma, ndatapoints):
    nclasses = means.shape[0]
    data = np.zeros((nclasses * ndatapoints, 5)) # cols 1-2 = inputs, cols 3-5 = desired output (for OR, AND, and XOR function)
    for c in range(0, nclasses):
        starti = c * ndatapoints
        endi = (c + 1) * ndatapoints
        data[starti:endi, 0:1] = means[c,0] + sigma * random.standard_normal((ndatapoints, 1))
        data[starti:endi, 1:2] = means[c,1] + sigma * random.standard_normal((ndatapoints, 1))
        if c > 0:
            data[starti:endi, 2] = 1 # OR
        if c == 3:
            data[starti:endi, 3] = 1 # AND
        if c == 1 or c == 2:
            data[starti:endi, 4] = 1 # XOR
    randvec = np.random.permutation(nclasses * ndatapoints)
    data = data[randvec,:]
    return data, randvec;

# plot the input for the OR-perceptron or the AND-perceptron
def plot_data(ndatapoints, data_output, figi):
    data = data_output[0]
    randvec = data_output[1]
    colors = np.concatenate((np.matlib.repmat(np.array([1, 0.5, 1]),ndatapoints,1),np.matlib.repmat(np.array([0.5, 1, 1]),ndatapoints,1),np.matlib.repmat(np.array([0.6, 1, 0.6]),ndatapoints,1),np.matlib.repmat(np.array([0.5, 0.5, 1]),ndatapoints,1)))
    colors = colors[randvec,:]
    plt.figure(figi)
    plt.scatter(data[:,0], data[:,1], c=colors, alpha=0.5)
    plt.axis('square')
    plt.xlabel('x1 (0 = green, 1 = red)')
    plt.ylabel('x2 (0 = small, 1 = large)')
    if figi == 1:
        plt.title('logical OR')
    elif figi == 2:
        plt.title('logical AND')
    return colors

# plot the input for the XOR-perceptron
def plot_data_XOR(colors, predictions_OR, predictions_AND, figi):
    plt.figure(figi)
    plt.scatter(predictions_OR,predictions_AND, c=colors, alpha=0.5) # complete this line yourself by providing the correct input arguments
    plt.axis('square')
    plt.xlabel('OR Input') # complete this line yourself by providing a label for the x axis
    plt.ylabel('AND Input') # complete this line yourself by providing a label for the y axis
    plt.title('Inputs of XOR Perceptron') # complete this line yourself by providing a title for the figure

# plot the decision boundary
def plot_boundary(weights, figi):
    b = weights[0]; w1 = weights[1]; w2 = weights[2]
    slope = -(b / w2) / (b / w1)
    y_intercept = -b / w2
    x = np.linspace(0,1,100)
    y = (slope * x) + y_intercept
    plt.figure(figi)
    plt.plot(x, y)
    plt.pause(0.4)

# predict output
def predict(inputs, weights):
    summation = np.dot(inputs, weights[1:]) + weights[0]
    if summation > 0:
      prediction = 1
    else:
      prediction = 0
    return prediction

# train the perceptron
def train(data, learning_rate, niterations, figi=0):
    training_inputs = data[:,0:2]
    labels = data[:,2]
    weights = 0.001 * random.standard_normal(data.shape[1])
    errors = np.zeros((data.shape[0], niterations))
    j = 0
    for _ in range(niterations):
        i = 0
        for inputs, label in zip(training_inputs, labels):
            prediction = predict(inputs, weights)
            weights[1:] += learning_rate * (label - prediction) * inputs
            weights[0] += learning_rate * (label - prediction)
            errors[i,j] = label - prediction
            if figi>0:
                plot_boundary(weights, figi)
            i += 1
        j += 1
    return weights, errors;

# test the perceptron
def test(data, weights):
    inputs_test = data[:,0:2]
    npredictions = data.shape[0]
    predictions = np.zeros(npredictions)
    for i in range(0, npredictions):
        predictions[i] = predict(inputs_test[i,:], weights)
    return predictions
