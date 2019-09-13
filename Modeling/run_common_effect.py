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



if len(sys.argv) > 1:
  i_LR = float(sys.argv[1])
  i_TE = int(sys.argv[2])
  i_L2 = float(sys.argv[3])
else:
  i_LR = 0.02
  i_TE = 500
  i_L2 = 0.0


total_part = 48
hrms = []
P_ams = []
N_ams = []
all_queries = []
all_stims = []

for part_number in np.arange(total_part):
  print("Participant number, ", part_number)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': i_TE,
                                   'test_epoch': 0,
                                   'L2': i_L2,
                                   'train_lr': i_LR,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 2,
                               'NONLIN' : 'tanh'},
            'train_blocks' : 30}
  
  
  expt = generative_causal.CommEff()
  expt_name = "Common_effect"
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  train_blocks = config['train_blocks']
  test_blocks = 20
  N_blocks = train_blocks + test_blocks
  
  # Optimization parameters
  train_epoch = config['optimization_params']['train_epoch']
  test_epoch = config['optimization_params']['test_epoch']
  L2 = config['optimization_params']['L2']
  train_lr = config['optimization_params']['train_lr']
  test_lr = config['optimization_params']['test_lr']
  
  # Network parameters
  OUT_DIM = 2
  INPUT_SIZE = 8 # A_pr, AC, B_pr, BC, C_pr, A_state, B_state, C_state
  NHID = config['network_params']['NHID']
  NONLIN = config['network_params']['NONLIN']
  
  storage_id = utils.make_id(config)
  
  # positive correlation vs negative correlation
  
  P_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  N_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  
  rational_model = expt.get_rationalmodel() 
  
  N_train_params =  expt.assign_params(train_blocks, condition = 'neg_corr_AB')
  P_train_params =  expt.assign_params(train_blocks, condition = 'pos_corr_AB')
  test_params =  expt.assign_params(test_blocks, condition = 'control')
  
  queries = np.random.binomial(1, 0.5, size = (N_blocks, 3))
  
  P_train_X = np.hstack((P_train_params, queries[:train_blocks]))
  P_train_X = torch.from_numpy(P_train_X)
  P_train_X = P_train_X.type(torch.FloatTensor)  
  
  N_train_X = np.hstack((N_train_params, queries[:train_blocks]))
  N_train_X = torch.from_numpy(N_train_X)
  N_train_X = N_train_X.type(torch.FloatTensor)  
  
  test_X = np.hstack((test_params, queries[train_blocks:]))
  test_X = torch.from_numpy(test_X)
  test_X = test_X.type(torch.FloatTensor)  
  
  
  # Create the data frames
  P_train_data = {'X': P_train_X,
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  N_train_data = {'X': N_train_X,
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  P_test_data = {'X': test_X,
               'y_hrm': None,
               'y_am': None,
               }
  
  N_test_data = {'X': test_X,
                 'y_hrm': None,
                 'y_am': None,
                 }
  
  # training models
  P_train_data = rational_model.train(P_train_data)
  
  P_approx_model.optimizer = optim.SGD(P_approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  P_approx_model.train(P_train_data, train_epoch)
  

  N_train_data = rational_model.train(N_train_data)
  N_approx_model.optimizer = optim.SGD(N_approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  N_approx_model.train(N_train_data, train_epoch)
    
  # testing models
  P_test_data = rational_model.test(P_test_data)
  
  P_approx_model.optimizer = optim.SGD(P_approx_model.parameters(), 
                                        lr=test_lr)
  P_test_data = P_approx_model.test(P_test_data, test_epoch)
    
  for key in P_test_data:
    if type(P_test_data[key]) is torch.FloatTensor:
      P_test_data[key] = P_test_data[key].numpy()
    else:
      P_test_data[key] = np.array(P_test_data[key])  

 
  N_test_data = rational_model.test(N_test_data)
  
  N_approx_model.optimizer = optim.SGD(N_approx_model.parameters(), 
                                        lr=test_lr)
  N_test_data = N_approx_model.test(N_test_data, test_epoch)
    
  for key in N_test_data:
    if type(N_test_data[key]) is torch.FloatTensor:
      N_test_data[key] = N_test_data[key].numpy()
    else:
      N_test_data[key] = np.array(N_test_data[key])  


  hrms.append(P_test_data['y_hrm'][:, 0])
  P_ams.append(P_test_data['y_am'][:, 0])
  N_ams.append(N_test_data['y_am'][:, 0])
  query_index = np.sum(np.array([1, 2, 4])*P_test_data['X'][:, -3:], axis = 1)
  all_queries.append(query_index)
  stim_index = np.sum(np.array([1, 3, 9, 27])*np.round(2*P_test_data['X'][:, :4]), axis = 1)
  all_stims.append(stim_index)
  
  
P_ams = np.reshape(np.array(P_ams), (-1))
N_ams = np.reshape(np.array(N_ams), (-1))
all_queries = np.reshape(np.array(all_queries), (-1))
all_stims = np.reshape(np.array(all_stims), (-1))
hrms = np.reshape(np.array(hrms), (-1))

# find true prob distributions over states
n_dist = []
p_dist = []
test_dist = []
#index = []

for state0 in product([0,1,2], repeat=4):
  state0 = tuple(np.asarray(state0)[::-1])
  state = state0 + (0,)
  p_dist.append(expt.find_probs(state, 'pos_corr_AB'))
  n_dist.append(expt.find_probs(state, 'neg_corr_AB'))
  test_dist.append(expt.find_probs(state, 'control'))
  #index.append(np.sum(np.array([1, 3, 9, 27])*np.asarray(state0), axis = 0))

n_dist = np.array(n_dist)
p_dist = np.array(p_dist)
test_dist = np.array(test_dist)
#index = np.array(index)
  

plot_data = {'P_ams': P_ams,
             'N_ams': N_ams,
             'hrms': hrms,
             'q': all_queries,
             'stim': all_stims,
             #'n_dist': n_dist,
             #'p_dist': n_dist,
             #'test_dist': test_dist
             }

utils.save_data(plot_data, name = storage_id + 'plot_data')

plt.plot(n_dist)
plt.plot(p_dist)
plt.plot(test_dist)
plt.show()
      