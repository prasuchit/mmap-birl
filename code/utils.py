import numpy as np
import math
import options
import numpy.matlib
from operator import mod
np.seterr(divide='ignore', invalid='ignore')

class trajNode:
    def __init__(self, s, a, parent):
        self.s = s
        self.a = a
        self.pair = str(s) + ' ' + str(a)
        self.parent = parent
    def __str__(self):
        s = ''
        s += 'sa: ' + str(self.s) + ', ' + str(self.a) + '\n'
        if self.parent is None:
            s += '  root not has no parent'
        else:
            s += '  parent: ' + str(self.parent.s) + ', ' + str(self.parent.a) + '\n'
        return s

def approxeq(V, oldV, EPS):
    return np.linalg.norm(np.reshape(V, len(V)) - np.reshape(oldV, len(oldV))) < EPS

def sampleWeight(name, nF, seed=None):
    np.random.seed(seed)
    w = np.zeros((nF, 1))
    w = np.random.rand(nF, 1)
    return w

def sid2info(sid, nS, nL, nG):

    tid = sid - 1;
    y3  = mod(tid, nG) + 1;
    tid = (tid - y3 + 1)/nG;
    y2  = mod(tid, nG) + 1;
    tid = (tid - y2 + 1)/nG;
    y1  = mod(tid, nG) + 1;
    tid = (tid - y1 + 1)/nG;
    myx = mod(tid, nL) + 1;
    tid = (tid - myx + 1)/nL;
    spd = mod(tid, nS) + 1;

def info2sid(spd, myx, y1, y2, y3, nS, nL, nG):

    sid = spd;
    sid = (sid - 1)*nL + myx;
    sid = (sid - 1)*nG + y1;
    sid = (sid - 1)*nG + y2;
    sid = (sid - 1)*nG + y3;

    
def convertW2R(weight, mdp):    
    mdp.weight = weight 
    reward = np.matmul(mdp.F, weight)
    reward = np.reshape(reward, (mdp.nStates, mdp.nActions), order='F')
    mdp.reward = reward
    return mdp

def QfromV(V, mdp): 
    nS = mdp.nStates
    nA = mdp.nActions
    Q = np.zeros((nS, nA))
    for a in range(nA):
        expected = np.matmul(np.transpose(mdp.transition[:, :, a]), V)
        Q[:, a] = mdp.reward[:, a] + mdp.discount * np.squeeze(expected)
    return Q

def find(arr, func):
    l = [i for (i, val) in enumerate(arr) if func(val)]
    if not l:
        return None
    else:
        return np.array(l).astype(int)

def getTrajInfo(trajs, mdp):
    nS = mdp.nStates
    nA = mdp.nActions

    trajInfo = options.trajInfo()
    trajInfo.nTrajs = trajs.shape[0]
    trajInfo.nSteps = trajs.shape[1]
    cnt = np.zeros((nS, nA))
    occupancy = np.zeros((nS, nA))
    nSteps = 0
    occlusions = []

    for m in range(trajInfo.nTrajs):
        for h in range(trajInfo.nSteps):
            s = trajs[m, h, 0]
            a = trajs[m, h, 1]
            if s == -1 and a == -1:
                occlusions.append([m, h])
            cnt[s, a] += 1  
            nSteps += 1
   
    trajInfo.occlusions = np.array(occlusions)

    N = np.count_nonzero(cnt)
    trajInfo.cnt = np.zeros((N, 3)).astype(int)
    i = 0
    for s in range(nS):
        for a in range(nA):
            if cnt[s, a] > 0:
                trajInfo.cnt[i, 0] = s
                trajInfo.cnt[i, 1] = a
                trajInfo.cnt[i, 2] = cnt[s, a]
                i += 1
    return trajInfo


def sampleNewWeight(dims, options, seed=None):
    np.random.seed(seed)
    lb = options.lb 
    ub = options.ub    
    if options.priorType == 'Gaussian':
        # w0 = options.mu + np.random.randn(dims, 1)*options.sigma  # Direct way to do it
        # for i in range(len(w0)):
        #     w0[i] = max(lb, min(ub, w0[i])) # Check to ensure weights are within bounds
        mean = np.ones(dims) * options.mu
        cov = np.eye(dims) * options.sigma
        w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))
    
    else:
        w0 = np.random.uniform(low=lb, high=ub, size=(dims,1))
    return w0