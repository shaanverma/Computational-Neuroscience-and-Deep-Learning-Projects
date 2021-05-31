#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:33:34 2021

@author: shaan
"""

'''
# load and examine the HMAX output
import pickle
import numpy as np

output = pickle.load( open('output.pkl', 'rb'))

s1 = output['s1']
n_sscales = len(s1) # s = spatial
s1_sscale1 = s1[0]
s1_sscale1.shape

# x y z vales in the diagram
x = np.shape(s1_sscale1)[2]
y = np.shape(s1_sscale1)[3]
z = np.shape(s1_sscale1)[1]

c1 = output['c1']

# m value
#m =np.shape(c1[6])[3])
m = np.shape(c1[6])[2]

# n value
#n =np.shape(c1[7])[3])
n = np.shape(c1[7])[2]


s2 = output['s2']
n_fscales = len(s2) # f = filter
s2_fscale1 = s2[7]
n_sscales = len(s2_fscale1)
s2_fscale1_sscale1 = s2_fscale1[7]
s2_fscale1_sscale1.shape

# s value
s = s2_fscale1_sscale1.shape[1]


# save answers
# insert x, y, z, m, n, s (integers), and layer name, operation A, operation B (strings) ...
#    ... as values in the dictionary below and save the dictionary as a text file using the code below
dict = {'x':x  , 'y':y  , 'z':z  , 'm':m  , 'n':n  , 's':s  , 'layer name':'C1'  , 'operation A':'?'  , 'operation B':'?'  } 
f = open('ps8_prob1.txt', 'w')
f.writelines( [str(dict['x'])+'\n', str(dict['y'])+'\n', str(dict['z'])+'\n', str(dict['m'])+'\n', str(dict['n'])+'\n', str(dict['s'])+'\n', dict['layer name']+'\n', dict['operation A']+'\n', dict['operation B']] )
f.close()
'''

# content of solution_code_ps8.py
def prob1_answers(i):
    f = open('ps8_prob1.txt', 'r')
    text = f.readlines()
    f.close() 
    
    return text[i]