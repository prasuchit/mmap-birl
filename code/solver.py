from models import mdp
import numpy as np
import math
import utils
import mdptoolbox
np.seterr(divide='ignore', invalid='ignore')

def policyIteration(mdp):

    MAX_ITERS = 10000
    EPS = 1e-12
    SHOW_MSG = False
    nS = mdp.nStates
    nA = mdp.nActions
    pi = mdptoolbox.mdp.PolicyIterationModified(mdp.transition.T, mdp.reward, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
    pi.run()
    Q = utils.QfromV(pi.V, mdp)
    piL = np.reshape(pi.policy, (nS, 1))
    H = evaluate(piL, mdp)
    return piL, pi.V, Q, H



def evaluate(piL, mdp):
    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))
    
    for a in range(nA):
        act_ind = utils.find(piL, lambda x: x == a)
        if act_ind is not None:
            Tpi[act_ind, :] = np.squeeze(np.transpose(mdp.transition[:, act_ind, a]))
    Epi = np.zeros((nS, nS * nA)).astype(int)
    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0] 
    return H