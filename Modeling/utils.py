import autograd.numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json, decimal
import pickle


# todo finish
def make_id(config, id = ''):
  key_list = sorted(config.iterkeys())
  for k in key_list:
    v = config[k]
    if isinstance(v, dict):
      id = make_id(v, id)
    else:
      id = id + str(k)+str(v)+'__'
  return id
      
      
class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)

#class DecimalEncoder(json.JSONEncoder):
    #def default(self, obj):
        #if isinstance(obj, D):
            #return float(obj)
        #return json.JSONEncoder.default(self, obj)
    
    
def def_npgaussian_lp(yval):
    #mean, std = yval[0,:D].view(1, -1), torch.exp(yval[0,-D:].view(1, -1))
    mean, std = yval[0], np.exp(yval[1])
    glp = lambda x : -(((x - mean)/std)**2 + np.log(2*np.pi*std**2))/2
    return glp

def def_npgaussian_gradlog(yval):
    #mean, std = yval[0,:D].view(1, -1), torch.exp(yval[0,-D:].view(1, -1))
    mean, std = yval[0], np.exp(yval[1])
    mgrad = lambda x : - (x-mean)/(std**2)
    lsdgrad = lambda x : - ((x-mean)/std)**2 + 1.0
    #lsdgrad = lambda x : 0.0
    glp = lambda x : np.array([mgrad(x), lsdgrad(x)])
    return glp

def find_KL(dist1, dist2):
    L = len(dist1)
    if L != len(dist2):
        raise ValueError("Distributions must be same length")
    kl = 0.0
    for i in xrange(L):
        #temp = dist1[i]*(np.log(dist1[i]) - np.log(dist2[i]))
        if (dist1[i] != 0 and dist2[i] != 0):
            #print kl
            kl += dist1[i]*(np.log(dist1[i]) - np.log(dist2[i]))

    #print kl
    return np.absolute(kl)


def gaussian_entropy(std):
    log_std = torch.log(std)
    norm = autograd.Variable(torch.Tensor([2*np.pi]))
    return 0.5 * len(std) * (1.0 + torch.log(norm)) + torch.sum(log_std)

def log_softmax(p, axis = 1):
  '''
  Currently assumes only 2 values in p
  '''
  #sums = np.sum(np.exp(p), axis = axis)
  #return np.log(np.exp(p)/sums[:, None])
  #print(np.round(p - np.logaddexp(p[0, 0], p[0, 1]) - np.log(np.exp(p)/sums[:, None])), 3)
  return p - np.logaddexp(p[0, 0], p[0, 1])

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def logit(p):
    return np.log(p) - np.log(1 - p)
        
        
def updates(array, N_trials, expt, prob = False):
    #prob = True
    if expt == "disc" : prob = True
    if prob:
        ret = np.array([inv_logit(array[i+1]) - inv_logit(array[i]) for i in np.arange(array.size - 1) \
                        if (i%N_trials != 0 and i%N_trials != N_trials-1)])
    else:
        ret = np.array([array[i+1] - array[i] for i in np.arange(array.size - 1) \
                        if (i%N_trials != 0 and i%N_trials != N_trials-1)])
    return ret



def get_binned(fbin, tbin, lim):
    #lim = 0.8
    num = 10
    bins = np.linspace(0,lim,num = num) + np.random.uniform(-0.05, 0.05)
    ind = np.digitize(fbin, bins = bins)
    y = []
    se = []
    x = []
    
    for i in np.arange(num):
        i += 1
        rvals = tbin[ind == i]
        if rvals.size:
            x.append(bins[i-1])
            y.append(np.mean(rvals))
            se.append(np.std(rvals))
    
    return (x, y, se)


    
def save_data(input_data, name):
    data = input_data.copy()
    fn = './data/' + name
    
    for key in data:
      if key != 'sub_exp' :
        data[key] = [float(x) for x in data[key].flatten()]
    with open(fn, 'wb') as outfile:
      json.dump(data, outfile, cls=DecimalEncoder)
    return
  

def load_data(name):
    fn = './data/' + name
    
    with open(fn, 'rb') as outfile:
      data = json.load(outfile)
    return data
    
def save_model(model, name):
    fn = './data/' + name
    torch.save(model.state_dict(), fn)
    print("Model Saved, {0}".format(fn))    
    return
    
def load_model(model, name):
    fn = './data/' + name
    model.load_state_dict(torch.load(fn))
    print("Model Loaded, {0}".format(fn))
    return

    
def plot_isocontours(ax, func, xlimits=[-20, 20], ylimits=[-20, 20], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, 30)
    ax.set_yticks([])
    ax.set_xticks([])

    
def plot_1D(ax, func, xlimits=[-20, 20], numticks=101):
    X = np.linspace(*xlimits, num=numticks)
    Y = func(X)
    plt.plot(X, Y)
    ax.set_yticks([])
    ax.set_xticks([])
    
def find_AR(bayes_post, subj_post, prior, randomize = False, clip = [-1000, 1000]):
  bayes_post = np.clip(bayes_post, 0.00000001, 0.99999999)
  subj_post = np.clip(subj_post, 0.00000001, 0.99999999)
  if randomize:
    which_urn = np.random.binomial(1, 0.5, bayes_post.shape)
    bayes_post, subj_post, prior = (which_urn*[bayes_post, subj_post, prior] + 
                                    (1.0 - which_urn)*[1.0 - bayes_post, 1.0 - subj_post, 1.0 - prior])
    
  B_post_odds = np.log(bayes_post/(1.0 - bayes_post))
  S_post_odds = np.log(subj_post/(1.0 - subj_post))
  BLLR = B_post_odds - np.log(prior/(1.0 - prior))
  SLLR = S_post_odds - np.log(prior/(1.0 - prior))
  exclusion = BLLR == 0.0
  ARs = np.empty(BLLR.shape)
  ARs[exclusion] = 1.0
  ARs[~exclusion] = SLLR[~exclusion]/BLLR[~exclusion]
  clip_mask = np.logical_or(ARs > clip[0], ARs > clip[1])
 
  return clip_mask, 1.0 - prior, ARs