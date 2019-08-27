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

    def assign_params(self, N_trials, condition, corr = None):
        strengths = np.array([0.2, 0.5, 0.8])
        #strengths = np.array([0.49, 0.5, 0.51])
        #p0 = np.array([0.33, 0.34, 0.33])
        
        p0 = np.array([0.4, 0.2, 0.4])
        p_r = np.array([0.1, 0.1, 0.8])
        p_l = np.array([0.8, 0.1, 0.1])
        p_m = np.array([0.2, 0.6, 0.2])
        p_control = np.array([0.1, 0.8, 0.1])
        
        self._params = (p_control, p0, p_r, p_l, p_m)
        
        if condition == 'control':
            #strengths = np.array([0.499, 0.5, 0.501])
            p0 = np.array([0.1, 0.8, 0.1])
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            B_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = np.random.choice(strengths, N_trials, p = p0)
            BC = np.random.choice(strengths, N_trials, p = p0)
            C_pr = 0.2*np.ones(N_trials)
            
        elif condition == 'pos_corr_AB':
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = np.random.choice(strengths, N_trials, p = p0)
            B_pr = []
            BC = []
            for t in np.arange(N_trials):
                
                if A_pr[t] < 0.5: p_pr = p_l
                if A_pr[t] > 0.5: p_pr = p_r
                if A_pr[t] == 0.5: p_pr = p_m
                B_pr.append(np.random.choice(strengths, 1, p = p_pr))
                
                
                if AC[t] < 0.5: pC = p_l
                if AC[t] > 0.5: pC = p_r
                if AC[t] == 0.5: pC = p_m
                BC.append(np.random.choice(strengths, 1, p = pC))
                
            B_pr = np.array(B_pr).squeeze()
            BC = np.array(BC).squeeze()
            C_pr = 0.2*np.ones(N_trials)
            
        elif condition == 'neg_corr_AB':
            p0 = np.array([0.4, 0.2, 0.4])
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = np.random.choice(strengths, N_trials, p = p0)
            B_pr = []
            BC = []
            for t in np.arange(N_trials):
                
                if A_pr[t] < 0.5: p_pr = p_r
                if A_pr[t] > 0.5: p_pr = p_l
                if A_pr[t] == 0.5: p_pr = p_m
                B_pr.append(np.random.choice(strengths, 1, p = p_pr))
                
                
                if AC[t] < 0.5: pC = p_r
                if AC[t] > 0.5: pC = p_l
                if AC[t] == 0.5: pC = p_m
                BC.append(np.random.choice(strengths, 1, p = pC))
                
            B_pr = np.array(B_pr).squeeze()
            BC = np.array(BC).squeeze()
            C_pr = 0.2*np.ones(N_trials)
            
        elif condition == 'strong_link':
            '''
            Not complete
            '''
            p = np.ones(3)*(1.0 - corr)/2.0
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            B_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = []
            BC = []
            C_pr = []
            for t in np.arange(N_trials):
                p_A = p.copy()
                p_A[strengths == A_pr[t]] = corr
                AC.append(np.random.choice(strengths, 1, p = p_A))
                
                p_B = p.copy()
                p_B[strengths == B_pr[t]] = corr
                BC.append(np.random.choice(strengths, 1, p = p_B))
                
                if A_pr[t] + B_pr[t] < 1.0: C_pr.append(strengths[-1])
                if A_pr[t] + B_pr[t] == 1.0: C_pr.append(strengths[1])
                if A_pr[t] + B_pr[t] > 1.0: C_pr.append(strengths[0])
                
                
            AC = np.array(AC).squeeze()
            BC = np.array(BC).squeeze()        
            C_pr = np.array(C_pr).squeeze()        
            
                        
        elif condition == 'weak_link':
            '''
            Not complete
            '''            
            p = np.ones(3)*(corr)/2.0
            A_pr = np.random.choice(strengths, N_trials, p = p0)
            B_pr = np.random.choice(strengths, N_trials, p = p0)
            AC = []
            BC = []
            C_pr = []
            for t in np.arange(N_trials):
                p_A = p.copy()
                p_A[strengths == A_pr[t]] = 1.0 - corr
                AC.append(np.random.choice(strengths, 1, p = p_A))
                
                p_B = p.copy()
                p_B[strengths == B_pr[t]] = 1.0 - corr
                BC.append(np.random.choice(strengths, 1, p = p_B))
                
                if A_pr[t] + B_pr[t] > 1.0: C_pr.append(strengths[-1])
                if A_pr[t] + B_pr[t] == 1.0: C_pr.append(strengths[1])
                if A_pr[t] + B_pr[t] < 1.0: C_pr.append(strengths[0])
                
                
            AC = np.array(AC).squeeze()
            BC = np.array(BC).squeeze()        
            C_pr = np.array(C_pr).squeeze()      
            
        else:
            raise ValueError('Choose a valid condition')
    
        
        return np.vstack((A_pr, AC, B_pr, BC, C_pr)).T
    
    def find_probs(self, state, condition):
        '''
        Currently tailored to pos/neg corr
        '''
        i_A_pr, i_AC, i_B_pr, i_BC, i_C_pr = state
        p_control, p0, p_r, p_l, p_m = self._params 
        
        
        if condition == 'pos_corr_AB':
            P_A_pr = p0[i_A_pr]
            P_AC = p0[i_AC]
            
            if i_A_pr < 1: P_B_pr_A_pr = p_l[i_B_pr]
            if i_A_pr > 1: P_B_pr_A_pr = p_r[i_B_pr]
            if i_A_pr == 1: P_B_pr_A_pr = p_m[i_B_pr]
            
            if i_AC < 1: P_BC_AC = p_l[i_BC]
            if i_AC > 1: P_BC_AC = p_r[i_BC]
            if i_AC == 1: P_BC_AC = p_m[i_BC]
            
        elif condition == 'neg_corr_AB':
            P_A_pr = p0[i_A_pr]
            P_AC = p0[i_AC]
            
            if i_A_pr < 1: P_B_pr_A_pr = p_r[i_B_pr]
            if i_A_pr > 1: P_B_pr_A_pr = p_l[i_B_pr]
            if i_A_pr == 1: P_B_pr_A_pr = p_m[i_B_pr]
            
            if i_AC < 1: P_BC_AC = p_r[i_BC]
            if i_AC > 1: P_BC_AC = p_l[i_BC]
            if i_AC == 1: P_BC_AC = p_m[i_BC]   
            
        elif condition == 'control':
            P_A_pr = p_control[i_A_pr]
            P_AC = p_control[i_AC]
            P_B_pr_A_pr = p_control[i_B_pr]
            P_BC_AC = p_control[i_BC]
            
        P_C_pr = 1.0
        
        prob = P_A_pr * P_AC * P_B_pr_A_pr * P_BC_AC * P_C_pr
        
        return prob

    
    def get_rationalmodel(self):
        return models.CommEffRational()
    


class CommCause(CBN):
    
    def data_gen(self, block_vals, N_trials, fixed = False, same_urn = False, return_urns = False, variable_ss = None):
        
        return 
   

    def assign_PL_CP(self, N_blocks, N_balls, alpha_post, alpha_pre):
        
        return 
    
    def get_rationalmodel(self, N_trials):
        
        return models.CommCauseRational(N_trials)
    
