#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 04:31:21 2021

@author: shaan
"""

# save the code below as ps10_main.py in the same folder as your solution code
# 
# input arguments for ps10
# outi:         output index
#               controls the output that is printed to the terminal
#               1 = cost before and after training
#               2 = training accuracy
#               3 = test accuracy 
# niterations:  number of iterations through the training set (epochs)
# alpha:        learning rate 
# monitor:      controls figure output
#               0 = no figures
#               1 = figures
#
# to test your solution code, set the input arguments below, and execute the following command from your terminal: run ps10_main.py

from solution_code_ps10 import ps10

ps10(
    outi = 3, # if you want to print the cost before and after training
    niterations = 2000, 
    alpha = 0.5,
    monitor = 1, 
)