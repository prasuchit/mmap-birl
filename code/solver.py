from models import mdp
import numpy as np
import math
import utils
np.seterr(divide='ignore', invalid='ignore')

def policyIteration(mdp):

    MAX_ITERS = 10000
    EPS = 1e-12
    SHOW_MSG = False

    nS = mdp.nStates
    nA = mdp.nActions

    oldpi = np.zeros((nS, 1)).astype(int)
    oldV = np.zeros((nS, 1)).astype(int)

    for iter in range(MAX_ITERS):
        [V, H] = evaluate(oldpi, mdp)
        Q = utils.QfromV(V, mdp)
        piL = np.reshape(np.argmax(Q, axis=1), (nS, 1))
        V = np.zeros((nS, 1))
        for i in range(nS):
            V[i, :] = Q[i, piL[i, :]]
        done = utils.approxeq(V, oldV, EPS) or np.array_equal(oldpi, piL)

        if done:
            break
        oldpi = piL
        oldV = V

    return piL, V, Q, H



def evaluate(piL, mdp):

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
    ################ I don't get this part #################   
    Epi = np.zeros((nS, nS * nA)).astype(int)   # Same dim as transition matrix
    idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS) # Write this down to understand why!
    for i in range(nS):
        Epi[i, idx[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]   # Least squares output of a linear function
    # H is the slope of the line fitting the x and y values in the lstsq func i/p
    # Check Map inference choi paper page 6 for the next line:
    # Given policy π, the reward optimality region is defined by Hπ = I − (IA −γT)(I −γTπ)^−1 * Eπ, and we can reuse the cached result if Hπ · Rnew ≤ 0. The
    V = np.matmul(H, w)
    #########################################################
    return V, H


