##########################
# AM4264 - PROBLEM SET 3 #
# SHAAN VERMA            #
# 250804514              #
##########################

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def lif_current_input( I_e=10*nA ):

    start_scope()

    # parameters
    taum   = 20*ms          # time constant
    g_L    = 10*nS          # leak conductance
    R_m    = 1/g_L 		    # membrane resistance
    E_l    = -70*mV         # leak reversal potential
    Vr     = E_l            # reset potential
    Vth    = -50*mV         # spike threshold
    Vs     = 20*mV          # spiking potential
    simTime = 1000*ms       # simulation time

    # Defining LIF equation
    eqs = '''
    dv/dt = ( E_l - v + R_m*I_e ) / taum  : volt (unless refractory)
    '''

    # create neuron
    G = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

    # initialize neuron
    G.v = E_l

    #Recording model state
    M = StateMonitor(G, 'v', record=True)

    #Recording number of spikes
    S = SpikeMonitor(G)

    #Running model simulation
    run(simTime)

    #Creating Plots
    '''
    plot(M.t/ms, M.v[0])
    xlabel('Time (ms)')
    ylabel('v');
    plt.show()
    '''

    #Calculating firing rate
    fr = S.num_spikes/(simTime/1000)

    #Calculating coefficient of variation
    ISIs = S.spike_trains()[0][1:]-S.spike_trains()[0][:-1]
    cv = np.std(ISIs)/np.mean(ISIs)

	# return values
    return fr,cv

def lif_poisson_input( v_e=10, v_i=10, w_e=0.1, w_i=0.4 ):
    start_scope()

    # parameters
    taum   = 20*ms          # time constant
    g_L    = 10*nS          # leak conductance
    E_l    = -70*mV         # leak reversal potential
    E_e    = 0*mV           # excitatory reversal potential
    tau_e  = 5*ms           # excitatory synaptic time constant
    E_i    = -80*mV         # inhibitory reversal potential
    tau_i  = 10*ms          # inhibitory synaptic time constant
    Nin    = 1000	        # number of synaptic inputs
    Ne     = int(0.8*Nin)   # number of excitatory inputs
    Ni     = int(0.2*Nin)   # number of inhibitory inputs
    Vr     = E_l            # reset potential
    Vth    = -50*mV         # spike threshold
    Vs     = 20*mV          # spiking potential
    simTime = 1000*ms       # simulation time

    # model equations
    eqs = '''
    dv/dt = ( E_l - v + g_e*(E_e-v) + g_i*(E_i-v) ) / taum  : volt (unless refractory)
    dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
    dg_i/dt = -g_i/tau_i  : 1  # inhibitory conductance (dimensionless units)
    '''

    # create neuron
    N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

    # initialize neuron
    N.v = E_l

    # create inputs
    Pe = PoissonGroup( 1, (v_e*Ne*Hz) )
    Pi = PoissonGroup( 1, (v_i*Ni*Hz) )

    # create connections
    synE = Synapses( Pe, N, 'w: 1', on_pre='g_e += w_e' ); synE.connect( p=1 );
    synI = Synapses( Pi, N, 'w: 1', on_pre='g_i += w_i' ); synI.connect( p=1 );


    # record model state
    M = StateMonitor( N, ('v','g_i'), record=True )
    S = SpikeMonitor( N )

    # run simulation
    run(simTime)

    # plot output
    '''
    plt.figure(figsize=(15,5)); plt.plot( M.t/ms, M.v[0] );
    plt.show()
    '''
    #Calculating firing rate
    fr = S.num_spikes/(simTime/1000)

    #Calculating coefficient of variation
    ISIs = S.spike_trains()[0][1:]-S.spike_trains()[0][:-1]
    cv = np.std(ISIs)/np.mean(ISIs)

    # return values
    return fr,cv

#############
# PROBLEM 1 #
#############
fr_current_input, cv_current_input = lif_current_input( I_e=10*nA )
fr_poisson_input, cv_poisson_input = lif_poisson_input( v_e=10, v_i=10, w_e=0.1, w_i=0.4 )

print(cv_current_input)
print(cv_poisson_input)

'''
When running the code for problem 1, the cv_current_input = 7.387627169094464e-15
cv_poisson_input = 1.1997797061987245. Since cv_current_input is larger, its stdev
must be higher and thus further from the mean when compared to cv_poisson_input.
Therby making the spikes more random and chaotic.

'''
###################################################
# PROBLEM 2 - Requires a couple of minutes to run #
###################################################

#Array of input rates
inp = np.arange(0,100,1)

#Array to hold output spikes
output = np.zeros(100)

count = 0

#populating arrays
for i in inp:
    fr, cv = lif_poisson_input(i,i)
    output[count] = fr
    count = count + 1

#ploting input rate and output spike rate
plt.plot(inp,output)
plt.xlabel("Input Rate")
plt.ylabel("Output Rate[spikes/sec]")
plt.show()

'''
Behaviour from graph:
As the input rate into the lif_poisson_input increases, the fr starts to increase as well.
B/n input values of 12-17 the firng rate is the highest, and then abruptly begins to decrease
as the input rate approaches 20. After 20, the firing rate remains 0.

Explaination:
As the input rate increase, the firing rate also increases which causes the resting potential
to become more negative meaning it becomes harder to reach the threshold potential for a spike.
This is why around an input rate of 20 onwards, the spike rate looks non-existent.
'''
