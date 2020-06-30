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

def sampleWeight(problem, nF, seed=None):
    
    np.random.seed(seed)
    w = np.zeros((nF, 1))
    # i = np.random.randint(0,5)
    i = 0
    if problem.name == 'gridworld':
        w = np.random.rand(nF, 1)
        # w = np.reshape(np.array([0.0330629367818319 , 0.768547209424092, 0.744437334981885, 0.574890390524460,
        #                         0.984201187783809, 0.885444972761834, 0.408018314905920, 0.177146367482004,
        #                         0.786364648105583, 0.519307375776462, 0.468017528895639, 0.955954804969700,
        #                         0.113075768967593, 0.482051172582913, 0.643050110155323, 0.975467398811565]), (nF,1))
    elif problem.name == 'highway':
        if i == 1:              # fast driver avoids collisions and prefers high speed
            w[:] = 0.01
            w[0]   = 0         # collision
            w[-1] = 0.5         # high-speed
        elif i == 2:            # safe driver avoids collisions and prefers right-most lane
            w[:] = 0.01
            w[0] = 0           # collision
            w[problem.nLanes] = 0.5 # right-most lane
        elif i == 3:            # erratic driver prefers collisions and high-speed
            w[:] = 0.01
            w[0] = 1            # collision
            w[-1] = 0.5         # high-speed
        else:
            w = np.random.rand(nF, 1)

    else:
        print("Unknown problem name!!")
        exit(0)

    return w
    
def convertW2R(weight, mdp):    
    mdp.weight = weight 
    reward = np.matmul(mdp.F, weight)
    reward = np.reshape(reward, (mdp.nStates, mdp.nActions), order='F')
    mdp.reward = reward
    return mdp

def sid2info(sid, nS, nL, nG):
# State id, num speeds, num lanes, num grids
    # print("SID to Info")
    # print(f"sid: {sid}| nS: {nS}| nL: {nL}| nG: {nG}")
    y = [None] * nL
    for i in range(nL-1,-1,-1):
        y[i] = int(mod(sid, nG))
        sid = (sid - y[i])/nG
    myx = int(mod(sid, nL))
    sid = int((sid - myx)/nL)
    spd = int(mod(sid, nS))
    # print(f"spd: {spd}| myx: {myx}| y: {y}")
    return spd, myx, y

def info2sid(spd, myx, y, nS, nL, nG):

    # print("Info to SID")
    sid = spd
    sid = (sid)*nL + myx
    for i in range(nL):
        sid = (sid)*nG + y[i]

    # print(f"spd: {spd}| myx: {myx}| y: {y}")
    # print(f"sid: {sid}| nS: {nS}| nL: {nL}| nG: {nG}")
    return sid

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
    occlusions = []

    for m in range(trajInfo.nTrajs):
        for h in range(trajInfo.nSteps):
            s = trajs[m, h, 0]
            a = trajs[m, h, 1]
            if s == -1 and a == -1:
                occlusions.append([m, h])
            else:
                cnt[s, a] += 1  
   
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
    # np.random.seed(seed)
    np.random.seed(None)
    lb = options.lb 
    ub = options.ub    
    if options.priorType == 'Gaussian':
        # w0 = options.mu + np.random.randn(dims, 1)*options.sigma  # Direct way to do it
        # for i in range(len(w0)):
        #     w0[i] = max(lb, min(ub, w0[i])) # Check to ensure weights are within bounds

        mean = np.ones(dims) * options.mu
        cov = np.eye(dims) * options.sigma
        w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))

        ''' Good weight(s) for testing 5 traj 10 steps 0 occl nGrid 4 nBlock 2 Gridworld ''' 
        # w0 = np.array([[-4.72298204e-01], [-2.39969794e-01], [-5.95197079e-05], [-1.93911446e-01]])
        # w0 = np.array([[-0.45573616], [ 0.53201201], [-0.1572897 ], [-0.52387681]])
        # w0 = np.array([[-0.3656997 ], [-0.02006794], [-0.25645092], [-0.08164048]]) # scipy weights
        ''' Good weight(s) for testing 5 traj 10 steps 1 occl nGrid 4 nBlock 2 Gridworld ''' 
        # w0 = np.array([[ 0.24116156], [-0.26847642], [ 0.06238525], [-0.08646028]])
    else:
        w0 = np.random.uniform(low=lb, high=ub, size=(dims,1))
    return w0