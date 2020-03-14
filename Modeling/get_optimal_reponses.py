import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product

import generative_causal
import utils

import matplotlib.pyplot as plt

import sys
import json

expt = generative_causal.CommEff()
rational_model = expt.get_rationalmodel() 

def get_prob_of(stim, A_state, B_state, C_state):
    
    #print(A_state, B_state, C_state)
    if np.isnan(A_state):
        prob = get_prob_of(stim, 0.0, B_state, C_state) + get_prob_of(stim, 1.0, B_state, C_state)
    elif np.isnan(B_state):
        prob = get_prob_of(stim, A_state, 0.0, C_state) + get_prob_of(stim, A_state, 1.0, C_state)
    elif np.isnan(C_state):
        prob = get_prob_of(stim, A_state, B_state, 0.0) + get_prob_of(stim, A_state, B_state, 1.0)
    else:
        prob = np.exp(rational_model.log_joint(A_pr, AC, B_pr, BC, C_pr, A_state, B_state, C_state)[0])
    
    #print(prob)
    return prob
    

def base10toN(num, base):
    """Change ``num'' to given base"""
    
    converted = []

    currentnum = num

    while np.any(currentnum):
        mod = currentnum % base
        currentnum = currentnum // base
        converted.append(mod)
    return np.array(converted)

stimulus = np.arange(81)
stim_weight_index = base10toN(stimulus, 3).T

# add dimension for fixed C_pr
stim_weight_index = np.hstack((stim_weight_index, np.zeros(81)[:, None]))

weight_map = {0:0.2, 1:0.5, 2:0.8}
stim_weights = np.vectorize(weight_map.get)(stim_weight_index)

X_states = np.array([np.array([0, np.nan, np.nan]),
                     np.array([1, np.nan, np.nan]),
                     np.array([0, 0, np.nan]),
                     np.array([1, 0, np.nan]),
                     np.array([0, 1, np.nan]),
                     np.array([1, 1, np.nan]),
                     np.array([0, np.nan, 0]),
                     np.array([1, np.nan, 0]),
                     np.array([0, np.nan, 1]),
                     np.array([1, np.nan, 1]),
                     np.array([0, 0, 0]),
                     np.array([1, 0, 0]),
                     np.array([0, 1, 0]),
                     np.array([1, 1, 0]),
                     np.array([0, 0, 1]),
                     np.array([1, 0, 1]),
                     np.array([0, 1, 1]),
                     np.array([1, 1, 1]),
                     ])
Y_states = X_states[:, [1, 0, 2]]
Z_states = X_states[:, [1, 2, 0]]

l_states = np.vstack((X_states, Y_states, Z_states))
corr_r_states = np.zeros_like(l_states)
corr_r_states[0::2] = l_states[1::2]
corr_r_states[1::2] = l_states[0::2]

optimal_responses = []

for stim in stim_weights:
    for (l_s, r_s) in zip(l_states, corr_r_states):
        A_pr, AC, B_pr, BC, C_pr = tuple(stim) 
        
        A_state, B_state, C_state = tuple(l_s)
        l_prob = get_prob_of(stim, A_state, B_state, C_state)
        
        A_state, B_state, C_state = tuple(r_s)
        r_prob = get_prob_of(stim, A_state, B_state, C_state)
        
        optimal_responses.append(np.round(100 * l_prob / (l_prob + r_prob)))
        
optimal_responses = np.array(optimal_responses)   
np.savetxt('data/optimal_responses.csv', optimal_responses, delimiter = ',')
    