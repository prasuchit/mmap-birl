import numpy as np
import math
import options
import utils2
import numpy.matlib
from scipy import sparse
import time
from operator import mod
np.seterr(divide='ignore', invalid='ignore')

def approxeq(V, oldV, EPS, useSparse):
    if useSparse:
        return sparse.linalg.norm(np.reshape(V, np.shape(V)) - np.reshape(oldV, np.shape(V))) < EPS
    else:
        return np.linalg.norm(np.reshape(V, len(V)) - np.reshape(oldV, len(V))) < EPS

def sampleWeight(problem, nF, seed=None):
    '''
    @brief Ground truth expert reward weights for different domains.
    '''
    np.random.seed(seed)
    w = np.zeros((nF, 1))
    i = 2   # behavior setter
    if problem.name == 'gridworld':
        if i == 0:  # Random behaviour
            w = np.random.rand(nF, 1)
        else:   # Forestworld weights
            w[0] = -0.5
            w[1] = -1
            w[2] = 1
    elif problem.name == 'highway':
        # weights are assigned 1 for collision, n for nlanes, 1 for high speed
        if i == 1:              # fast driver avoids collisions and prefers high speed
            w[0] = -1        # collision
            w[-1] = 0.1         # high-speed
        elif i == 2:            # safe driver avoids collisions and prefers right-most lane
            w[0] = -1           # collision
            w[problem.nLanes] = 0.1 # right-most lane
            w[-1] = -0.001 # Slight penalty for fast speed

        elif i == 3:            # erratic driver prefers collisions and high-speed
            w[0] = 1            # collision
            w[-1] = 0.1         # high-speed
        else:
            w = np.random.rand(nF, 1)

    elif problem.name == 'sorting':
        '''Features: Good onion back on conveyor;
                     Bad onion back on conveyor;
                     Good onion in bin;
                     Bad onion in bin; 
                     Claim new onions;
                     Pick if unknown;'''
        w = np.reshape(np.array([1,-1,-1,1,1.5,0.5]), (nF,1))
    else:
        print("Unknown problem name!!")
        exit(0)

    return w
    
def convertW2R(weight, mdp):    
    '''
    @brief Converts weights into reward function by multiplying 
    with the corresponding feature vector.
    '''
    if mdp.useSparse:
        mdp.weight = sparse.csr_matrix(weight)
    else:
        mdp.weight = weight
    reward = np.dot(mdp.F, mdp.weight)
    reward = np.reshape(reward, (mdp.nStates, mdp.nActions), order='F')
    mdp.reward = reward
    if mdp.useSparse:
        mdp.reward = sparse.csr_matrix(mdp.reward)
        for a in range(mdp.nActions):
            mdp.rewardS[a] = sparse.csr_matrix(mdp.reward[:, a])
    return mdp

def sid2info(sid, nS, nL, nG):
    '''
    FOR HIGHWAY PROBLEM
    @brief - Given state id, this func converts it to the state variable values. 
    '''
    y = [None] * nL
    for i in range(nL-1,-1,-1):
        y[i] = int(mod(sid, nG))
        sid = (sid - y[i])/nG
    myx = int(mod(sid, nL))
    sid = int((sid - myx)/nL)
    spd = int(mod(sid, nS))
    return spd, myx, y

def info2sid(spd, myx, y, nS, nL, nG):
    '''
    FOR HIGHWAY PROBLEM
    @brief - Given the variable values making up a state, this converts it into state id 
    '''
    sid = spd
    sid = (sid)*nL + myx
    for i in range(nL):
        sid = (sid)*nG + y[i]
    return sid

def QfromV(V, mdp): 
    '''
    @brief Function name self-explanatory.
    '''
    nS = mdp.nStates
    nA = mdp.nActions
    if mdp.useSparse:
        Q = sparse.csr_matrix(np.zeros((nS, nA)))
        for a in range(nA):
            expectedS = np.dot(np.transpose(mdp.transitionS[a][:, :]), V)
            Q[:, a] = mdp.rewardS[a] + mdp.discount * np.squeeze(expectedS)
        
    else:
        Q = np.zeros((nS, nA))
        for a in range(nA):
            expected = np.matmul(np.transpose(mdp.transition[:, :, a]), V)
            Q[:, a] = mdp.reward[:, a] + mdp.discount * np.squeeze(expected)
    return Q

def find(arr, func):
    '''
    @brief Return indices of array where function is true with array value.
    '''
    l = [i for (i, val) in enumerate(arr) if func(val)]
    if not l:
        return None
    else:
        return np.array(l).astype(int)

def getOrigTrajInfo(trajs, mdp):
    ''' 
    @brief Compute occupancy measure, occlusion info and empirical policy for trajectories. 
    '''

    nS = mdp.nStates
    nA = mdp.nActions

    trajInfo = options.trajInfo()
    trajInfo.nTrajs = trajs.shape[0]
    trajInfo.nSteps = trajs.shape[1]
    occlusions, cnt, occupancy, allOccNxtSts = utils2.processOccl(trajs, nS, nA, trajInfo.nTrajs, trajInfo.nSteps, mdp.discount, mdp.transition)
   
    trajInfo.occlusions = np.array(occlusions)
    trajInfo.allOccNxtSts = np.array(allOccNxtSts)
    """
    piL = np.nan_to_num(cnt / np.matlib.repmat(cnt.sum(axis=1).reshape((nS, 1)), 1, nA))
    mu = (cnt.sum(axis=1) / trajInfo.nSteps).reshape((nS, 1))
    occupancy = occupancy / trajInfo.nTrajs
    nSnA = nS*nA
    occupancy = occupancy.reshape((nSnA, 1), order='F')
    
    # trajInfo.v = np.matmul(np.transpose(mdp.reward.reshape((nSnA, 1))), occupancy).squeeze()
    trajInfo.pi = piL   # empirical estimate of policy
    trajInfo.mu = mu    # state visitation
    trajInfo.occupancy = occupancy  # discounted state-action frequency
    trajInfo.featExp = np.matmul(np.transpose(mdp.F), trajInfo.occupancy)    # feature expectation
    """
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

def getTrajInfo(trajs, mdp):
    ''' 
    @brief Compute occupancy measure and empirical policy for trajectories. 
    '''
    nS = mdp.nStates
    nA = mdp.nActions

    trajInfo = options.trajInfo()
    trajInfo.nTrajs = trajs.shape[0]
    trajInfo.nSteps = trajs.shape[1]  
    cnt = np.zeros((nS, nA))
    occupancy = np.zeros((nS, nA))
    for m in range(trajInfo.nTrajs):
        for h in range(trajInfo.nSteps):
            s = trajs[m, h, 0]
            a = trajs[m, h, 1]
            if -1 not in trajs[m, h, :]:
                cnt[s, a] += 1      
                occupancy[s, a] += math.pow(mdp.discount, h)
    """
    piL = np.nan_to_num(cnt / np.matlib.repmat(cnt.sum(axis=1).reshape((nS, 1)), 1, nA))
    mu = (cnt.sum(axis=1) / trajInfo.nSteps).reshape((nS, 1))
    occupancy = np.divide(occupancy,trajInfo.nTrajs)
    nSnA = nS*nA
    occupancy = occupancy.reshape((nSnA, 1), order='F')
    
    # trajInfo.v = np.matmul(np.transpose(mdp.reward.reshape((nSnA, 1))), occupancy).squeeze()
    trajInfo.pi = piL   # empirical estimate of policy
    trajInfo.mu = mu    # state visitation
    trajInfo.occupancy = occupancy  # discounted state-action frequency
    trajInfo.featExp = np.dot(np.transpose(mdp.F), trajInfo.occupancy)   # feature expectation
    """
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
    '''
    @brief Samples weights from a multivariate distribution.
    '''
    np.random.seed(seed)
    # np.random.seed(None)
    lb = options.lb 
    ub = options.ub    
    if options.priorType == 'Gaussian':
        # w0 = options.mu + np.random.randn(dims, 1)*options.sigmasq  # Direct way to do it
        # for i in range(len(w0)):
        #     w0[i] = max(lb, min(ub, w0[i])) # Check to ensure weights are within bounds

        mean = np.ones(dims) * options.mu
        cov = np.eye(dims) * options.sigmasq
        w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))
    else:
        w0 = np.random.uniform(low=lb, high=ub, size=(dims,1))
    return w0