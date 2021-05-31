'''
AM4264 - PROBLEM SET 1
NAME: Shaan Verma
STUDENT#: 250804514
DATE: 2021-01-21
'''

import numpy as np
import matplotlib.pyplot as plt

'''
Problem 1 - Runs two simulations of the logistic map
            One with x0 and the other with x0 + eps.
'''
def problem_1( a, x0, nsteps, print_step):

    # Initializing empty arrays
    x = np.zeros(nsteps)
    x2 = np.zeros(nsteps)

    # Setting the first element in the arrays
    x[0] = x0
    x2[0] = x0 + np.finfo(float).eps

    #Running simulations
    for i in np.arange(1, nsteps):
        x[i] = a * x[i - 1] * (1 - x[i - 1])
        x2[i] = a * x2[i - 1] * (1 - x2[i - 1])

    # plot results
    '''
    plt.plot(x, label="x")
    plt.plot(x2, label="x2")
    plt.legend()
    plt.title('First and Second Simulation')
    plt.xlabel('Time steps')
    plt.ylabel('x')

    plt.figure()
    plt.title('Change in First and Second Simulation')
    plt.xlabel('Time steps')
    plt.ylabel('|x - x2|')

    plt.plot(np.abs(x-x2))
    plt.show()
    '''

    print((np.abs(x - x2))[print_step])

'''
Problem 2: Comparing numerical with analytical solution.
'''
def problem_2(a, x0, nsteps, print_step):

    # Initializing empty arrays
    x = np.zeros(nsteps)
    x2 = np.zeros(nsteps)

    # Setting the first element in the arrays
    x[0] = x0
    x2[0] = x0

    # Running simulations
    for i in np.arange(1, nsteps):
        x[i] = a * x[i - 1] * (1 - x[i - 1])
        x2[i] = 0.5*(1 - np.cos((2**i)*np.arccos(1-(2*x2[0]))))

    # plot results
    '''
    plt.plot(x, label="Numerical")
    plt.plot(x2, label="Analytical")
    plt.legend()
    plt.title('Numerical and Analytical Simulation')
    plt.xlabel('Time steps')
    plt.ylabel('x')

    plt.figure()
    plt.title('Change in Numerical and Analytical Simulation')
    plt.xlabel('Time steps')
    plt.ylabel('|x - x2|')

    plt.plot(np.abs(x - x2))
    plt.show()
    '''

    print(((np.abs(x-x2))[print_step] < 1e-5))