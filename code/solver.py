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
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))

    for a in range(nA):
        act_ind = utils.find(piL, lambda x: x == a)
        if act_ind is not None:
            Tpi[act_ind, :] = np.squeeze(
                np.transpose(mdp.transition[:, act_ind, a]))
    Epi = np.zeros((nS, nS * nA)).astype(int)
    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS)
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]

    return H


def policyIteration(mdp):

    """ This is the naive way to do policy
    iteration. Since we have a toolbox available,
    this function is currently unused"""

    oldpi = np.zeros((nS, 1)).astype(int)
    oldV = np.zeros((nS, 1)).astype(int)

    for iter in range(MAX_ITERS):
        [V, H] = evaluate(oldpi, mdp)
        Q = utils.QfromV(V, mdp)
        # Sec 2.2 Theorem 2 Eq 3 Algo for IRL
        piL = np.reshape(np.argmax(Q, axis=1), (nS, 1))
        V = np.zeros((nS, 1))
        for i in range(nS):
            V[i, :] = Q[i, piL[i, :]]
        done = utils.approxeq(V, oldV, EPS) or np.array_equal(oldpi, piL)

        if done:
            break
        oldpi = piL
        oldV = V

        # if mdp.useSparse:
        #     oldV = sparse.csr_matrix(mdp.nStates, 1)
        # else:
        #     oldV = zeros(mdp.nStates, 1)

    return piL, V, Q, H


def evaluate(piL, mdp):

    """This function is being called 
    from policy iteration function.
    Hence it's currently unused."""
    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))

    for a in range(nA):
        act_ind = utils.find(piL, lambda x: x == a)
        if act_ind is not None:
            Tpi[act_ind, :] = np.squeeze(
                np.transpose(mdp.transition[:, act_ind, a]))
    Epi = np.zeros((nS, nS * nA)).astype(int)
    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS)
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]
    V = np.matmul(H, w)
    return V, H