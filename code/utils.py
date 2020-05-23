import numpy as np
import math
# import scipy.io as sio
import options
import numpy.matlib
np.seterr(divide='ignore', invalid='ignore')

class trajNode:
    def __init__(self, s, a, parent):
        self.s = s
        self.a = a
        self.pair = str(s) + ' ' + str(a)
        self.parent = parent
#################################################################################
# The init and str here are not being used currently.                           #
# My guess is they are needed to print out the nodes of the graph               #
# after analysis of the bayesian network using the computations.                #
#################################################################################
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
    w = np.zeros((nF, 1))
    return w

def convertW2R(weight, mdp):    # Converting weights into rewards
    print("Updating mdp weights...")
    mdp.weight = weight # This is adjusted as we move through the statespace
    reward = np.matmul(mdp.F, weight)   # IRL algorithm originally considers the 
                                        # reward as a weighted linear combination of features
    reward = np.reshape(reward, (mdp.nStates, mdp.nActions), 'F')   # Making reward a 144*4 matrix
    mdp.reward = reward
    return mdp

def QfromV(V, mdp): 
    nS = mdp.nStates
    nA = mdp.nActions
    Q = np.zeros((nS, nA))
    for a in range(nA): # Eq 1 Sec 2.1 Choi kim paper
        expected = np.matmul(np.transpose(mdp.transition[:, :, a]), V)
        # Section 2.2, Theorem 1, eq 2 Algorithms for IRL
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
            cnt[s, a] += 1  # Empirical occupancy matrix
            occupancy[s, a] += math.pow(mdp.discount, h)  # discounted state occupancy (visitation frequency)
            nSteps += 1
    # Check Choi-Kim MAP paper page 3 starting to find this eq for mu and v

    occupancy = occupancy / trajInfo.nTrajs # This is alpha from eq 1 of your choi-kim writeup
    # Above line is part of eq 6 in your notes of choi-kim's paper
    nSnA = nS*nA    # 144*4
    reward_reshaped = mdp.reward.reshape((nSnA, 1))    # Reshaping reward matrix from 144*4 to 576*1 linear vector
    reward_reshaped_transposed = np.transpose(reward_reshaped)  # 1*576 reward vector
    occupancy_reshaped = occupancy.reshape((nSnA, 1))  # Reshaping occupancy matrix from 144*4 to 576*1 linear vector
    # Check section 5.1.4 page 22 starting of Survey of IRL to see how to calculate V(s)
    # trajInfo.v = np.matmul(reward_reshaped_transposed, occupancy_reshaped).squeeze()  # 1*576 * 576*1 = 1*1   # State specific cumulative reward values V(s)
    trajInfo.v = np.matmul(reward_reshaped_transposed, occupancy_reshaped)  # 1*576 * 576*1 = 1*1   # State specific cumulative reward values V(s)
    # The above eq is eq 6 in your choi-kim notes
    # empirical_occupancy = np.sum(cnt).reshape((nS, 1)) # Original code
    empirical_occupancy = cnt.sum(axis=1).reshape((nS, 1))  # This is irrespective of the action and only wrt state,
    # By summing axis=1 (across columns for each row) values, we add the freq values for all actions in a state
    # Whereas without adding that the freq corresponds to an action in that state
    # Check eq 8 and 9 in you choi-kim paper notes 
    empirical_occupancy_distributed = np.matlib.repmat(empirical_occupancy, 1, nA)  # Denominator on RHS of pi(s,a) from the paper
    trajInfo.pi = np.nan_to_num(cnt / empirical_occupancy_distributed)   # Policy
    # trajInfo.mu = (np.sum(cnt) / nSteps).reshape((nS, 1)) # Original code
    trajInfo.mu = (cnt.sum(axis=1) / nSteps).reshape((nS, 1)) # Mu(s) is Emperical state visitation freq for that state s
    trajInfo.occupancy = occupancy
    feature_transpose = np.transpose(mdp.F)
    feature_reshaped = (feature_transpose).reshape((nS,nS))
    # if(((mdp.F).reshape(nS,nS) == (np.transpose(mdp.F)).reshape((nS,nS))).all()):
    #     print("They're the same!")
    # else:
    #     print("You were wrong!")
    trajInfo.featExp = np.matmul(feature_reshaped,trajInfo.mu)   # mdp.F is calculated in gridworld.py
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
    lb = options.lb    # lower bound
    ub = options.ub    # upper bound
    # dims - dimensions; Depends on number of features.
    if options.priorType == 'Gaussian':
        mean = np.ones(dims) * options.mu
        cov = np.eye(dims) * options.sigma
        w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))
        # Sampling a random value using the mean and covariance from a normal distribution
    else:
        w0 = np.random.uniform(low=lb, high=ub, size=(dims))
        # Sampling a random value from a uniform distribution b/w lb and ub
    return w0