from models import mdp
import numpy as np
import math
import utils
import mdptoolbox
from scipy import sparse
np.seterr(divide='ignore', invalid='ignore')


def piMDPToolbox(mdp):

    MAX_ITERS = 10000
    EPS = 1e-12
    SHOW_MSG = False
    nS = mdp.nStates
    nA = mdp.nActions
    if mdp.useSparse:
        pi = mdptoolbox.mdp.PolicyIterationModified(np.transpose(
            mdp.transitionS), mdp.rewardS, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
    else: 
        pi = mdptoolbox.mdp.PolicyIterationModified(np.transpose(
            mdp.transition), mdp.reward, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
    pi.run()
    Q = utils.QfromV(pi.V, mdp)
    piL = np.reshape(pi.policy, (nS, 1))
    H = evalToolbox(piL, mdp)

    return piL, pi.V, Q, H


def evalToolbox(piL, mdp):

    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    if mdp.useSparse:
        I = sparse.eye(nS)
        Tpi = sparse.csr_matrix(np.zeros((nS, nS)))
        for a in range(nA):
            state_idx = utils.find(piL, lambda x: x == a)   # state(s) index of action that matches policy action   # state index of action that matches policy action
            if state_idx is not None:
                Tpi[idx, :] = np.squeeze(np.transpose(mdp.transition[:, state_idx, a]))
        Epi = sparse.csr_matrix(np.zeros((nS, nS * nA)).astype(int))
        state_idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS) # Scaled s*a index
        for s in range(nS):
            Epi[s, state_idx[s]] = 1
    else:
        I = np.eye(nS)
        Tpi = np.zeros((nS, nS))

        for a in range(nA):
            state_idx = utils.find(piL, lambda x: x == a)   # state(s) index of action that matches policy action
            if state_idx is not None:
                Tpi[state_idx, :] = np.squeeze(np.transpose(mdp.transition[:, state_idx, a]))
        Epi = np.zeros((nS, nS * nA)).astype(int)
        state_idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS) # Scaled s*a index
        for s in range(nS):
            Epi[s, state_idx[s]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]

    return H


def policyIteration(mdp):

    """ 
    @brief This is the naive way to do policy iteration. Since we have a toolbox available,
    this function is currently only used for sparse mdp implementation
    """

    MAX_ITERS = 10000
    EPS = 1e-12
    SHOW_MSG = False
    nS = mdp.nStates
    nA = mdp.nActions
    oldpi = np.zeros((nS, 1)).astype(int)
    if mdp.useSparse:
        oldV = sparse.csr_matrix(np.zeros((nS, 1)).astype(int))
    else:
        oldV = np.zeros((nS, 1)).astype(int)

    for iter in range(MAX_ITERS):
        [V, H] = evaluate(oldpi, mdp)
        Q = utils.QfromV(V, mdp)
        # Sec 2.2 Theorem 2 Eq 3 Algo for IRL
        if mdp.useSparse:    
            piL = np.reshape(np.array(sparse.csr_matrix.argmax(Q, axis=1).squeeze()), (nS, 1))
            V = sparse.csr_matrix(np.zeros((nS, 1)))
        else:
            piL = np.reshape(np.argmax(Q, axis=1), (nS, 1))
            V = np.zeros((nS, 1))
        for i in range(nS):
            V[i] = Q[i, piL[i]]
        done = utils.approxeq(V, oldV, EPS, mdp.useSparse) or np.array_equal(oldpi, piL)

        if done:
            break
        oldpi = piL
        oldV = V

    return piL, V, Q, H


def evaluate(piL, mdp):

    """ 
    @brief This function is being called from policy iteration function.
    Hence it's currently only used for sparse mdp implementation 
    """

    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    if mdp.useSparse:
        I = sparse.eye(nS)
        Tpi = sparse.csr_matrix((nS, nS))
        for a in range(nA):
            state_idx = utils.find(piL, lambda x: x == a)   # state(s) index of action that matches policy action
            if state_idx is not None:
                Tpi[state_idx, :] = np.squeeze(np.transpose(mdp.transition[:, state_idx, a]))
        Tpi = sparse.csr_matrix(Tpi)
        Epi = sparse.csr_matrix(np.zeros((nS, nS * nA)).astype(int))
        state_idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS) # Scaled s*a index
        for s in range(nS):
            Epi[s, state_idx[s]] = 1
    else:
        I = np.eye(nS)
        Tpi = np.zeros((nS, nS))

        for a in range(nA):
            state_idx = utils.find(piL, lambda x: x == a)   # state(s) index of action that matches policy action
            if state_idx is not None:
                Tpi[state_idx, :] = np.squeeze(
                    np.transpose(mdp.transition[:, state_idx, a]))
        Epi = np.zeros((nS, nS * nA)).astype(int)
        state_idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS) # Scaled s*a index
        for s in range(nS):
            Epi[s, state_idx[s]] = 1

    if mdp.useSparse:
        H = np.linalg.lstsq((I - mdp.discount * Tpi).todense(), np.dot(Epi, mdp.F).todense())[0]
        H = sparse.csr_matrix(H)
    else:
        H = np.linalg.lstsq((I - mdp.discount * Tpi), np.matmul(Epi, mdp.F))[0]
    
    V = np.dot(H, w)
    return V, H