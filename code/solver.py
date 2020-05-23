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
        piL = np.reshape(np.argmax(Q, axis=1), (nS, 1)) # Sec 2.2 Theorem 2 Eq 3 Algo for IRL
        V = np.zeros((nS, 1))
        for i in range(nS):
            V[i, :] = Q[i, piL[i, :]]
        done = utils.approxeq(V, oldV, EPS) or np.array_equal(oldpi, piL)

        if done:
            break
        oldpi = piL
        oldV = V
    # V has just one best value for a state. Q has values for all actions in a state.
    return piL, V, Q, H



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
    # temp = piL * nS
    # idxReshapeDim1 = piL * nS + np.arange(0, nS).reshape((nS , 1))
    idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS) 
    for i in range(nS):
        Epi[i, idx[i]] = 1  # Setting expectation of that state and action corresponding to the policy

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]   # Least squares output of a linear function
    
    # IA = np.stack((I,I,I,I))
    # T_disc = mdp.discount*mdp.transition
    # temp1 = IA - T_disc.T
    # # inverse = np.linalg.inv(I - mdp.discount * Tpi)
    # temp2 = np.matmul(temp1,H)
    # # temp3 = np.matmul(Epi, mdp.F)
    # # temp4 = I - np.matmul(temp2,temp3)
    # temp5 = I - temp2
    # print("Reached here")

    # H is the slope of the line fitting the x and y values in the lstsq func i/p
    # Check Map inference choi paper page 6 for the next line:
    # Given policy π, the reward optimality region is defined by Hπ = I − (IA −γT)(I −γTπ)^−1 * Eπ, and we can reuse the cached result if Hπ · Rnew ≤ 0. The
    V = np.matmul(H, w)
    return V, H