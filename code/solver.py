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
    # ################### Original code ##########################################
    # oldpi = np.zeros((nS, 1)).astype(int)
    # oldV = np.zeros((nS, 1)).astype(int)

    # for iter in range(MAX_ITERS):
    #     [V, H] = evaluate(oldpi, mdp)
    #     Q = utils.QfromV(V, mdp)
    #     piL = np.reshape(np.argmax(Q, axis=1), (nS, 1)) # Sec 2.2 Theorem 2 Eq 3 Algo for IRL
    #     V = np.zeros((nS, 1))
    #     for i in range(nS):
    #         V[i, :] = Q[i, piL[i, :]]
    #     done = utils.approxeq(V, oldV, EPS) or np.array_equal(oldpi, piL)

    #     if done:
    #         break
    #     oldpi = piL
    #     oldV = V
    # V has just one best value for a state. Q has values for all actions in a state. 
    # print(piL.squeeze())
    # print(V.squeeze())
    # print(Q.squeeze())
    ###################################################################################
    return piL, pi.V, Q, H



def evaluate(piL, mdp):
    # piL is just the policy
    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))
    
    for a in range(nA):
        # For lambda function explanation, see: https://stackoverflow.com/a/890188
        idx = utils.find(piL, lambda x: x == a) # Returns an np array of values that match the action value in the piL array
        if idx is not None:
            Tpi[idx, :] = np.squeeze(np.transpose(mdp.transition[:, idx, a]))   # For all next state of the current state given by index value in idx
    Epi = np.zeros((nS, nS * nA)).astype(int)   # Same dim as transition matrix
    idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS) 
    for i in range(nS):
        Epi[i, idx[i]] = 1  # Setting expectation of that state and action corresponding to the policy

    # H equation is provided at the end of the supplementary material of Choi and Kim's 
    # MAP for BIRL paper, under theorem 2.
    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]   # Least squares output of a linear function
    # V = np.matmul(H, w)
    # return V, H
    return H