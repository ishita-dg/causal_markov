import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import generative_causal
import utils

import matplotlib.pyplot as plt

import sys
import json



if len(sys.argv) > 1:
  total_part = int(sys.argv[1])
else:
  total_part = 20

hrms = []
ams = []

for part_number in np.arange(total_part):
  print("Participant number, ", part_number)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 50,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.007,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 1,
                               'NONLIN' : 'rbf'},
            'N_balls' : 20,
            'alpha_pre' : 1.0, 
            'train_blocks' : 200,
            'N_trials' : 4}
  
  # Run results for Correction Prior (CP)
  
  expt = generative_causal.CommEff()
  expt_name = "Common_effect"
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = config['N_trials']
  
  train_blocks = config['train_blocks']
  test_blocks = 200
  N_blocks = train_blocks + test_blocks
  
  N_balls = config['N_balls']
  
  # Optimization parameters
  train_epoch = config['optimization_params']['train_epoch']
  test_epoch = config['optimization_params']['test_epoch']
  L2 = config['optimization_params']['L2']
  train_lr = config['optimization_params']['train_lr']
  test_lr = config['optimization_params']['test_lr']
  
  # Network parameters -- single hidden layer MLP
  # Can also adjust the nonlinearity
  OUT_DIM = 2
  INPUT_SIZE = 5 #data, lik1, lik2, prior, N
  NHID = config['network_params']['NHID']
  NONLIN = config['network_params']['NONLIN']
  
  storage_id = utils.make_id(config)
  
  # Informative data vs uninformative data
  
  approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  rational_model = expt.get_rationalmodel(N_trials) 
  
  train_block_vals =  expt.assign_PL_CP(train_blocks, N_balls, alpha_post = 0.27, alpha_pre = config['alpha_pre'])
  train_X = expt.data_gen(train_block_vals, N_trials, N_balls)
  test_block_vals =  expt.assign_PL_CP(test_blocks, N_balls, alpha_post = 0.27, alpha_pre = config['alpha_pre'])
  test_X = expt.data_gen(test_block_vals, N_trials)
  
  # Create the data frames
  train_data = {'X': train_X,
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  
  test_data = {'X': test_X,
               'y_hrm': None,
               'y_am': None,
               }
  
  
  # training models
  train_data = rational_model.train(train_data)
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  approx_model.train(train_data, train_epoch)
  #utils.save_model(approx_model, name = storage_id + 'trained_model')
  
  # testing models
  test_data = rational_model.test(test_data)
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=test_lr)
  test_data = approx_model.test(test_data, test_epoch, N_trials)
  #utils.save_model(approx_model, name = storage_id + 'tested_model')
  
  for key in test_data:
    if type(test_data[key]) is torch.FloatTensor:
      test_data[key] = test_data[key].numpy()
    else:
      test_data[key] = np.array(test_data[key])
      
  #utils.save_data(test_data, name = storage_id + 'test_data')
  if True :
    hrms.append(test_data['y_hrm'][:, 1])
    ams.append(test_data['y_am'][:, 1])
  else:
    print("**********reject this participant")
  
  
ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms), (-1))
which_urn = np.random.binomial(1, 0.5, ams.shape)
ams = ams*which_urn + (1 - which_urn)*(1.0 - ams)
hrms = hrms*which_urn + (1 - which_urn)*(1.0 - hrms)

notnan = np.logical_not(np.isnan(ams))
ams = ams[notnan]
hrms = hrms[notnan]

# Plotting
f, (ax1, ax2) = plt.subplots(1, 2)
jump = 0.05
bins = np.arange(0.0, 1.0, jump)
x0 = bins + jump/2.0
p_x = np.arange(-0.1, 1.1, jump)
Y_means = []
Y_vars = []
x = []
digitized = np.digitize(hrms, bins)
for d in np.arange(len(bins)):
  if not np.isnan(np.mean(ams[digitized == d+1])):
    Y_means.append(np.mean(ams[digitized == d+1]))
    Y_vars.append(np.var(ams[digitized == d+1]))
    x.append(x0[d])

x = np.array(x)
ax1.scatter(x, Y_means, label = 'Beta = 0.27')
#ax1.scatter(hrms, ams)
ax1.set_xlim([-0.1, 1.1])
ax1.set_ylim([-0.1, 1.1])
ax1.plot([0.0, 1.0], [0.0, 1.0], c = 'k')
ax1.axvline(0.5, c = 'k')
ax1.set_title("Conservatism effect")

cons_fit_params = np.polyfit(x, Y_means, 1)
cons_fit = np.poly1d(cons_fit_params)
ax1.plot(p_x, cons_fit(p_x), c = 'r')

ax2.scatter(x, Y_vars, label = 'Beta = 0.27')
ax2.set_title("Variance effect")

var_fit_params = np.polyfit(x, Y_vars, 2)
var_fit = np.poly1d(var_fit_params)
ax2.plot(p_x, var_fit(p_x), c = 'r')
ax2.plot(p_x, np.mean(Y_vars)*np.ones(len(p_x)), linestyle = ':')

#plt.legend()
plt.show()
plt.savefig('figs/Conservatism_' + storage_id +'.pdf')
    

plot_data = {'x': x,
             'var': np.array(Y_vars),
             'mean': np.array(Y_means),
             'all_ams': ams,
             'all_hrms': hrms}

utils.save_data(plot_data, name = storage_id + 'plot_data')
        
        
