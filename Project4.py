#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM4264 Problem Set 4
@author: shaan
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

'''
N is the number of components in the vector
trials is the number of times thhe simulation is run
'''
def problem_1(N, trials):
    holder = []

    for i in range(trials):

        #Creating random arrays
        v1 = np.random.rand(N)
        v2 = np.random.rand(N)

        #Normalizing the vectors
        v1_unit = v1/np.linalg.norm(v1)
        v2_unit = v2/np.linalg.norm(v2)

        #Taking the dot product
        value = np.dot(v1_unit, v2_unit)

        #Adding the values to the end of the array
        holder.append(value)

    return np.mean(holder)

#UNcomment the following code to test problem_1
'''
meanVal = []
NVal = list(range(0,10001))

for i in NVal:
    meanVal.append(problem_1(i, 10))

plt.plot(NVal,meanVal)
plt.xlabel("N Value")
plt.ylabel("Mean")
plt.show()
'''


def problem_2():
    N = 1000
    taum = 10*ms
    taupre = 20*ms
    taupost = taupre
    Ee = 0*mV
    vt = -54*mV
    vr = -60*mV
    El = -74*mV
    taue = 5*ms
    F = 15*Hz
    wmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= wmax
    dApre *= wmax

    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    input = PoissonGroup(N, rates=F)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='linear')
    S = Synapses(input, neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w
                        Apre += dApre
                        w = clip(w + Apost, 0, wmax)''',
                 on_post='''Apost += dApost
                         w = clip(w + Apre, 0, wmax)''',
                 )
    S.connect()
    S.w = 'wmax'

    mon = StateMonitor(S, 'w', record=True)
    # the resulting ndarray will now have the shape (N,timesteps),
    # which will allow accessing the time evolution of individual
    # synapses (e.g. mon.w[0] will be the time evolution of the first
    # input synapse)spike_trains = spike_mon.spike_trains()

    #Changed from input to neurons
    s_mon = SpikeMonitor(neurons)

    run(10*second, report='text')

    subplot(311)
    plot(S.w / wmax, '.k')
    ylabel('Weight / wmax')
    xlabel('Synapse index')
    subplot(312)
    hist(S.w / wmax, 20)
    xlabel('Weight / wmax')
    subplot(313)
    plot(mon.t/second, mon.w[0]/wmax) # timecourse of synaptic weight for first input synapse
    xlabel('Time (s)')
    ylabel('Weight / wmax')

    spike_trains = s_mon.spike_trains()

    #Holds the first two seconds of spikes
    first = []

    #Holds the last two seconds of spikes
    last = []

    #first 2 seconds of simulation
    for i in spike_trains[0]:
        if(i<=2*second):
            first.append(i)

    #Last 2 seconds of simulation
    for j in spike_trains[0]:
        if(i>=8*second):
            last.append(j)

    #Calculating Cv values
    ISIs_f = np.diff(first)
    cv_f = np.std(ISIs_f)/np.mean(ISIs_f)

    ISIs_l = np.diff(last)
    cv_l = np.std(ISIs_l)/np.mean(ISIs_l)

    #cv_f is the first 2sec, cv_l is the last 2 sec
    return (cv_f, cv_l)

print(problem_2())
