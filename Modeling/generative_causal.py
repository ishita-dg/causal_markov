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


class CBN ():
    
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
        
        return models.MLP_disc(INPUT_SIZE, NUM_LABELS, nhid, None, Causal.VI_loss_function_grad, nonlin, stronginit)
    
    
        #return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = nn.KLDivLoss())
        
        
class CommEff(CBN):
    
    def data_gen(self, block_vals, N_trials, fixed = False, same_urn = False, return_urns = False, variable_ss = None):
        
        '''
        TODO: Change to noisy_or
        '''
        ps, l1s, l2s = block_vals
        
        N_blocks = len(ps)        
    
        draws = np.empty(shape = (N_trials*N_blocks, 1))
        lik1s = np.empty(shape = (N_trials*N_blocks, 1))
        lik2s = np.empty(shape = (N_trials*N_blocks, 1))
        pris = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        true_urns = np.empty(shape = (N_trials*N_blocks))
        
        
        for i, (p,l1,l2) in enumerate(zip(ps, l1s, l2s)):
            
            if same_urn:
                urn_b = np.random.binomial(1, p, 1)*np.ones(N_trials)
            else:
                urn_b = np.random.binomial(1, p, N_trials)
             
            draws_b = []
            delNs = [0]
            delN = 0
            for urn in urn_b:
                if urn:
                    lik = l1
                else:
                    lik = l2
                
                if variable_ss is not None:                        
                    draws_b.append(0.0)
                    heads = np.random.binomial(variable_ss[i], lik, 1)
                    tails = variable_ss[i] - heads
                    delN = (heads - tails)[0]
                    delNs.append(delN)            
                        
                else:
                    draw = np.random.binomial(1, lik, 1)
                    draws_b.append(draw)
                    #success = draw*(1-urn) + (1-draw)*urn
                    if same_urn:
                        delN += (2*draw - 1)[0]
                    else:
                        delN = 0
                    delNs.append(delN)
            
            
            if fixed: draws_b = np.ones(N_trials)
            delNs = np.array(delNs, dtype = np.float)
            if variable_ss is not None:
                if variable_ss[i] == 0:
                    Ns[i*N_trials : (i+1)*N_trials, 0] = 0.0
                else: 
                    Ns[i*N_trials : (i+1)*N_trials, 0] = delNs[1]/variable_ss[i]
            else:
                Ns[i*N_trials : (i+1)*N_trials, 0] = 1.0*delNs[:-1] / N_trials              
            draws[i*N_trials : (i+1)*N_trials, 0] = draws_b
            lik1s[i*N_trials : (i+1)*N_trials, 0] = l1 * np.ones(N_trials)
            lik2s[i*N_trials : (i+1)*N_trials, 0] = l2 * np.ones(N_trials)            
            pris[i*N_trials : (i+1)*N_trials, 0] = p * np.ones(N_trials)
            true_urns[i*N_trials : (i+1)*N_trials] = urn_b
            
        
        if variable_ss is not None:
            normalized_ss = 1.0*variable_ss / max(variable_ss)
            X = np.hstack((draws, lik1s, lik2s, pris, Ns, 
                           normalized_ss.reshape((-1, 1))
                           ))
        else:
            X = np.hstack((draws, lik1s, lik2s, pris, Ns, np.ones(shape = (N_trials*N_blocks, 1))))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        
        
        if return_urns:
            return X, true_urns
        
        return X
            

   

    def assign_PL_CP(self, N_blocks, N_balls, alpha_post, alpha_pre):
        
            
        posts = np.random.beta(alpha_post, alpha_post, N_blocks)
        pres = np.random.beta(alpha_pre, alpha_pre, N_blocks)#0.5*np.ones(N_blocks)
        priors = []
        likls = []
        
        for pre, post in zip(pres, posts):
            if np.abs(pre - post) > 0.5:
                pre = 1.0 - pre
            x = (post*(1.0 - pre))/(pre*(1.0 - post))
            edit = x / (1.0 + x)
            
            ep = np.clip(np.round(edit*N_balls), 1, N_balls - 1)
            pp = np.clip(np.round(pre*N_balls), 1, N_balls - 1)
            if (np.random.uniform() > 0.0):
                priors.append(pp*1.0 / N_balls)
                likls.append([ep*1.0 / N_balls, 1.0 - ep*1.0 / N_balls])
            else:
                priors.append(ep*1.0 / N_balls)
                likls.append([pp*1.0 / N_balls, 1.0 - pp*1.0 / N_balls])                
                
        
        return np.array(priors).reshape((-1,1)), np.array(likls).reshape((-1,2))[:, 0], np.array(likls).reshape((-1,2))[:, 1] 
    
    
    def get_rationalmodel(self, N_trials):
        
        return models.CommEffRational(N_trials)
    


class CommCause(CBN):
    
    def data_gen(self, block_vals, N_trials, fixed = False, same_urn = False, return_urns = False, variable_ss = None):
        
        return 
   

    def assign_PL_CP(self, N_blocks, N_balls, alpha_post, alpha_pre):
        
        return 
    
    def get_rationalmodel(self, N_trials):
        
        return models.CommCauseRational(N_trials)
    
