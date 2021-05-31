"""
AM 4264 Intro to Neural Networks
Lab #2
Shaan Verma
250804514
"""

'''
Importing scientific computing packages
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice  # import this to slice time within the "for" loop

'''
Parameters
'''
Rm = 1e6  # resistance (ohm)
Cm = 2e-8  # capacitance (farad)
taum = Rm * Cm  # time constant (seconds)
Vr = -.060  # resting membrane potential (volt)
Vreset = -.070  # membrane potential after spike (volt)
Vth = -.050  # spike threshold (volt)
Vs = .020  # spiking potential (volt)

'''
Defining Time axis
'''
dt = .001  # simulation time step (seconds)
T = 1.0  # total time to simulate (seconds)
time = np.linspace(dt, T, int(T / dt))  # vector of timepoints we will simulate

'''
Helper functions
'''
def initialize_simulation():
    # zero-pad membrane potential vector 'V' and spike vector 'spikes'
    V = np.zeros(time.size)  # preallocate vector for simulated membrane potentials
    spikes = np.zeros(time.size)  # vector to denote when spikes happen - spikes will be added after LIF simulation
    V[0] = Vr  # set first time point to resting potential
    return V, spikes


def logistic_map(a, x0, nsteps):
    # function to simulate logistic map:
    # x_{n+1} = a * x_n * (1-x_n)
    x = np.zeros(nsteps)
    x[0] = x0
    for ii in range(1, nsteps):
        x[ii] = a * x[ii - 1] * (1 - x[ii - 1])
    return x


def plot_potentials(time, V, timeSpikes):
    # plots membrane potential (V) against time (time), and marks spikes with red markers (timeSpikes)
    plt.show()
    plt.plot(time, V, 'k', timeSpikes, np.ones(timeSpikes.size) * Vs, 'ro')
    plt.ylabel('membrane potential (mV)')
    plt.xlabel('time (seconds)')


def check_solutions(result, solution_filename):
    # check solutions against provided values
    solution = np.load(solution_filename)
    if (np.linalg.norm(np.abs(result - solution)) < 0.1):
        print('\n\n ---- problem solved successfully ---- \n\n')


def integrate_and_fire(V, spikes, i, Ie):
    # function to integrate changes in local membrane potential and fire if threshold reached
    # V - vector of membrane potential
    # spikes - spike marker vector
    # i - index (applied to V and spikes) for current time step
    # Ie - input current at this time step (scalar of unit amp)

    # 1: calculate change in membrane potential (dV)
    dV = dt * (Vr - V[i - 1] + Rm * Ie) / taum
    # 2: integrate over given time step (Euler method)
    V[i] = V[i - 1] + dV
    # 3: does the membrane potential exceed threshold (V > Vth)?
    if (V[i] > Vth):
        V[i] = Vreset
        spikes[i] = 1
    return V, spikes  # output the membrane potential vector and the {0,1} vector of spikes


def problem_1():
    # ////////////////////////////////////
    #  problem 1 - step current input //
    # //////////////////////////////////
    #
    # Implement a leaky integrate and fire (LIF) neuron with parameters given
    # above.
    #
    # Create a current input which:
    #       - starts at 0 A
    #       - steps up to 15 nA at stim_time[0]
    #       - steps down to 0 A at stim_time[1]
    #
    # Output:
    # Plot the resulting simulated membrane potential of the LIF neuron.
    #

    # problem-specific parameters
    stim_time = [.2, .8]  # time (seconds) when current turns ON and turns OFF

    V, spikes = initialize_simulation()  # initialize simulation

    for i, t in islice(enumerate(time), 1, None):
        if t > stim_time[0] and t < stim_time[1]:
            ie = 1.5e-8
        else:
            ie = 0
        V, spikes = integrate_and_fire(V, spikes, i, ie)

    # add spikes to create membrane potential waveforms
    V[spikes == 1] = Vs

    # plot membrane potential
    plot_potentials(time, V, time[spikes == 1])
    plt.title('P1 : step current input')

    # output:
    V_prob1 = V
    return V_prob1


def problem_2():
    # //////////////////////////////////////////
    # problem 2 - oscillating current input //
    # ////////////////////////////////////////
    # Use the LIF implementation from problem 1.
    # Create a current input which:
    #       - starts at 0 A
    #       - oscillates with a cosine of amplitude 20 nA at stim_time[0]
    #       - stops oscillating and returns to 0 A at stim_time[1]
    #
    # output:
    # Plot the resulting simulated membrange potential of the LIF, and save the
    # membrane potential in a vector named "V_prob2".

    # problem-specific parameters
    f = 10  # Hz
    phase = np.pi

    V, spikes = initialize_simulation()  # initialize simulation
    stim_time = [.2, .8]  # time (seconds) when current turns ON and turns OFF

    #Holder for current values
    ie = np.zeros(len(time))

    for i, t in islice(enumerate(time), 1, None):
        if t > stim_time[0] and t < stim_time[1]:
            ie[i] = 2e-8*np.cos((2*np.pi*f*t) - phase)
        else:
            ie[i] = 0
        V, spikes = integrate_and_fire(V, spikes, i, ie[i])

    # add spikes
    V[spikes == 1] = Vs

    # PLOT membrane potential
    plot_potentials(time, V, time[spikes == 1])
    plt.title('Problem 2: Oscillating current input')

    # output:
    V_prob2 = V
    return V_prob2


def problem_3():
    # ////////////////////////////////////////////////////
    # problem 3 - scan across oscillation frequencies //
    # //////////////////////////////////////////////////
    # Using previous problem's simulation (i.e. oscillating current input),
    # run a simulation per frequency stored in the variable "freq".
    #
    # output: plot the results, and then save the number of spikes generated in
    # each run in a variable named "nspikes_prob3".

    # problem-specific parameters
    freq = np.linspace(15, 50, (50 - 15) + 1)  # Hz
    phase = np.pi
    oscillation_amplitude = 4e-8  # amps

    #Initializing sum variable to hold total spikes
    sum = 0

    #Holder for current values
    ie = np.zeros(len(time))

    stim_time = [.2, .8]

    # initialize array
    nSpikes = np.zeros(freq.size)

    # iterate each freq
    for j, f in enumerate(freq):
        V, spikes = initialize_simulation()  # initialize simulation
        for i, t in islice(enumerate(time), 1, None):
            if t > stim_time[0] and t < stim_time[1]:
                ie[i] = oscillation_amplitude * np.cos((2 * np.pi * f * t) - phase)
            else:
                ie[i] = 0
            V, spikes = integrate_and_fire(V, spikes, i, ie[i])
        #Adding up the spikes
        for z in spikes:
            if z == 1.0:
                sum = sum + 1
        nSpikes[j] = sum
        sum = 0

    # PLOT number of spikes per frequency
    plt.show()
    plt.plot(freq, nSpikes, 'ko')
    plt.title('Problem 3: Scan across oscillating frequencies')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('# of spikes')

    return nSpikes


def problem_4():
    # //////////////////////////////////////////
    # problem 4 - fluctuating current input //
    # ////////////////////////////////////////
    # Use the LIF implementation from simulation 1.
    # Create a current input using a logistic map in the chaotic regime (a=4).
    # Add an additional current step starting at stim_time[0] and ending at
    # stim_time[1]
    #
    # output:
    # Plot the resulting simulated membrange potential of the LIF, and save the
    # membrane potential in a vector named "V_prob4".

    # Parameters:
    a = 4
    lm_x0 = 0.6
    lm_range = 5e-8  # parameters for logistic map (ensure the mean is 0 by subtracting 0.5)
    current_step = 1e-8
    stim_time = [.2, .8]

    values = (logistic_map(a, lm_x0, time.size) - 0.5) * lm_range
    V, spikes = initialize_simulation()  # initialize simulation

    for i, t in islice(enumerate(time), 1, None):
        if t > stim_time[0] and t < stim_time[1]:
            ie = values[i] + current_step
        else:
            ie = 0
        V, spikes = integrate_and_fire(V, spikes, i, ie)

    # add spikes
    V[spikes == 1] = Vs

    # PLOT membrane potential
    plot_potentials(time, V, time[spikes == 1])
    plt.title('Problem 4: Chaotic input')

    # output:
    V_prob4 = V
    return V_prob4



'''
###############################
#       Driver function       #
# --------------------------- #
# Test Cases for all problems #
###############################
'''
def driver():
    print("Problem #1")
    check_solutions(problem_1(), "problem1.npy")
    plt.figure()

    print("Problem #2")
    check_solutions(problem_2(), "problem2.npy")
    plt.figure()

    print("Problem #3")
    check_solutions(problem_3(), "problem3.npy")
    plt.figure()

    print("Problem #4")
    check_solutions(problem_4(), "problem4.npy")
    plt.show()

'''Uncomment to run test cases'''
#driver()
