import numpy as np
import torch
#import torch.distributions as dists
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import def_npgaussian_lp
from utils import def_npgaussian_gradlog
from utils import log_softmax
import models
from torch.distributions import Categorical
from torch.distributions import Normal
reload(models)

torch.manual_seed(1)


class CBN():
    
    @staticmethod
    def Exact_VI_loss_function(yval, target):
        p = torch.exp(target)/torch.sum(torch.exp(target))
        q = torch.exp(yval)
        
        KL = torch.mm(torch.log(q) - torch.log(p), q.view(-1,1))
            
        return KL

    
    @staticmethod
    def VI_loss_function(yval, target):
        nsamps = 50
        ELBOs = []
        logq = yval.view(1,-1)
        logp = target.view(1,-1)
        #logp = torch.log(torch.exp(target)/torch.sum(torch.exp(target)).view(1,-1))
        
        count = 0
        dist_q = Categorical(torch.exp(logq))
        while count < nsamps:
            s = dist_q.sample().type(torch.LongTensor)            
            #ELBO = logp.index_select(1,s)
            ELBO = logq.index_select(1,s) * (logp.index_select(1,s) - logq.index_select(1,s)/2)
            #print(p.data.numpy() - q.data.numpy(),s.data.numpy(),ELBO.data.numpy())
            ELBOs.append(ELBO)
            count += 1
            
        
        qentropy = 0
        #qentropy = torch.sum(torch.exp(logq) * logq)
        loss = -(torch.mean(torch.cat(ELBOs)) - qentropy)
            
    
        return loss
    
    
    @staticmethod
    def VI_loss_function_grad(yval, target):
        '''
        TODOs:
        1. check dimensionality of q for replications in while loop
-            s = np.random.multinomial(1, q, 1)
-            onehot = np.reshape(s, (-1,1))
-            ELBO_grad += np.reshape(q-s, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
+            onehot = np.zeros(2)
+            s = np.reshape(np.random.multinomial(1, np.reshape(q, (-1))), (-1,1))
+            onehot[s[0][0]] = 1
+            ELBO_grad += np.reshape(q[0]-onehot, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
        '''
        nsamps = 50
        ELBOs = []
        logq = log_softmax(yval.view(1,-1).data.numpy())
        logp = target.view(1,-1).data.numpy()
        q = np.exp(logq)[0]
        L = logp.shape[1]
        count = 0
        ELBO_grad = 0
        while count < nsamps:
            s = np.random.multinomial(1, q, 1)
            onehot = np.reshape(s, (-1,1))
            ELBO_grad += np.reshape(q-s, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
            count += 1
            
        
        grad = ELBO_grad/count
    
        return autograd.Variable(torch.Tensor(grad).type(torch.FloatTensor).view(1,-1)   )

       
    
    def get_approxmodel(self, NUM_LABELS, INPUT_SIZE, nhid, nonlin, stronginit = False):
        
        return models.MLP_disc(INPUT_SIZE, NUM_LABELS, nhid, None, CBN.VI_loss_function_grad, nonlin, stronginit)
    
    
        #return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = nn.KLDivLoss())
        
        
class CommEff(CBN):   

    def assign_params(self, N_trials, condition, corr = 0.8):
        strengths = np.array([0.2, 0.5, 0.8])
        #strengths = np.array([0.49, 0.5, 0.51])
        p0 = np.array([0.33, 0.34, 0.33])
        C_pr = 0.1*np.ones(N_trials)
        
        if condition == 'control':
            strengths = np.array([0.49, 0.5, 0.51])
            A_pr = np.random.choice(strengths, N_trials)
            B_pr = np.random.choice(strengths, N_trials)
            AC = np.random.choice(strengths, N_trials)
            BC = np.random.choice(strengths, N_trials)
            
        elif condition == 'pos_corr':
            p = np.ones(3)*(1.0 - corr)/2.0
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = np.random.choice(strengths, N_trials, p = p0)
            B_pr = []
            BC = []
            for t in np.arange(N_trials):
                p_pr = p.copy()
                p_pr[strengths == A_pr[t]] = corr
                B_pr.append(np.random.choice(strengths, 1, p = p_pr))
                
                p_C = p.copy()
                p_C[strengths == AC[t]] = corr
                BC.append(np.random.choice(strengths, 1, p = p_C))
            B_pr = np.array(B_pr).squeeze()
            BC = np.array(BC).squeeze()
            
        elif condition == 'neg_corr':
            p = np.ones(3)*(corr)/2.0
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = np.random.choice(strengths, N_trials, p = p0)
            B_pr = []
            BC = []
            for t in np.arange(N_trials):
                p_pr = p.copy()
                p_pr[strengths == A_pr[t]] = 1.0 - corr
                B_pr.append(np.random.choice(strengths, 1, p = p_pr))
                
                p_C = p.copy()
                p_C[strengths == AC[t]] = 1.0 - corr
                BC.append(np.random.choice(strengths, 1, p = p_C))
            B_pr = np.array(B_pr).squeeze()
            BC = np.array(BC).squeeze()
    
        
        return np.vstack((A_pr, AC, B_pr, BC, C_pr)).T
    
    def get_rationalmodel(self):
        
        return models.CommEffRational()
    


class CommCause(CBN):
    
    def data_gen(self, block_vals, N_trials, fixed = False, same_urn = False, return_urns = False, variable_ss = None):
        
        return 
   

    def assign_PL_CP(self, N_blocks, N_balls, alpha_post, alpha_pre):
        
        return 
    
    def get_rationalmodel(self, N_trials):
        
        return models.CommCauseRational(N_trials)
    
