import models
import math
import numpy as np
import sys
import utils
from scipy import sparse

np.set_printoptions(threshold=sys.maxsize)
np.seterr(divide='ignore', invalid='ignore')

def init(gridSize=12, blockSize=2, noise=0.3, discount=0.99, useSparse = 0):
    mdp = models.mdp()

    nS = gridSize * gridSize 
    nA = 4
    # nF = int(math.pow(gridSize/blockSize, 2))
    nF = 3
    T = np.zeros((nS, nS, nA))
    for y in range(gridSize):
        for x in range(gridSize):
            s = loc2s(x, y, gridSize)
            # x , y = s2loc(s, gridSize)
            ns = np.zeros(nA).astype(int)
            ns[0] = loc2s(x, y + 1, gridSize) # N  # Creating a list of possible states you
            ns[1] = loc2s(x + 1, y, gridSize) # E  # could reach by performing the available
            ns[2] = loc2s(x - 1, y, gridSize) # W  # list of actions in each state.
            ns[3] = loc2s(x, y - 1, gridSize) # S

            for a in range(nA):
                for a2 in range(nA):
                    T[ns[a2], s, a] = T[ns[a2], s, a] + (noise / nA) # Adding noise per action to each transition 
                T[ns[a], s, a] = T[ns[a], s, a] + (1 - noise)    # Rest of the prob given to the intended action
    
    F = np.zeros((nS, nF)).astype(int)
    for y in range(gridSize):
        for x in range(gridSize):
            s = loc2s(x, y, gridSize)
            i = math.floor(x / blockSize)
            j = math.floor(y / blockSize)
            f = loc2s(i, j, int(gridSize/blockSize)) 
            # F[s, f] = 1
            if s == 6:
                F[s,0] = 1
            elif s == 11:
                F[s,1] = 1
            elif s == 15:
                F[s,2] = 1


    # for i in range(nS):
    #     np.savetxt("state"+str(i)+".csv", np.round(T[:,i,:], 4), delimiter=",")

    start = np.ones((nS, 1))
    start = start / (np.sum(start) - 3) # Excluding the prob of the 3 below states.
    start[6] = 0    # Pond
    start[11] = 0   # Bad state
    start[15] = 0   # Goal state
    # forest_cover = [3, 4, 11]
    mdp.name = 'gridworld'
    mdp.gridSize = gridSize
    mdp.blockSize = blockSize
    mdp.nStates = nS
    mdp.nActions = nA
    mdp.nFeatures = nF
    mdp.discount = discount
    mdp.start = start
    mdp.transition = T
    mdp.F = np.tile(F, (nA, 1))
    mdp.weight = np.zeros((nF,1))
    mdp.reward = np.reshape(np.matmul(mdp.F,mdp.weight), (nS, nA))
    mdp.useSparse = useSparse
    mdp.sampled = False
    # mdp.forest_cover = forest_cover

    if mdp.useSparse:
        mdp.transitionS = {}
        mdp.rewardS = {}
        mdp.F      = sparse.csr_matrix(mdp.F)
        mdp.weight = sparse.csr_matrix(mdp.weight)
        mdp.start  = sparse.csr_matrix(mdp.start) 
        for a in range(mdp.nActions):
            mdp.transitionS[a] = sparse.csr_matrix(mdp.transition[:, :, a])
            mdp.rewardS[a] = sparse.csr_matrix(mdp.reward[:, a])

    return mdp

def loc2s(x, y, gridSize):  # Location x,y value to state num mapping
    x = max(0, min(gridSize - 1, x))
    y = max(0, min(gridSize - 1, y))
    return y * gridSize + x


def s2loc(s, gridSize):
    """
    @brief: Convert a state int into the corresponding coordinate.

    s: State id.
    -> (x, y) int tuple.
    """
    return (s % gridSize, s // gridSize)

def neighbouring(i, k, gridSize):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: state id.
        k: state id.
        -> bool.
        """
        i = s2loc(i, gridSize)
        k = s2loc(k, gridSize)
        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1