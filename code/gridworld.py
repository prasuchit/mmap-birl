import models
import math
import numpy as np
import sys

# np.set_printoptions(threshold=sys.maxsize)
np.seterr(divide='ignore', invalid='ignore')

def init(gridSize=12, blockSize=2, noise=0.3, discount=0.99):
    mdp = models.mdp()
    mdp.name = ''

    nS = gridSize * gridSize    # Number of states for 12 x 12 grid = 144
    nA = 4  # Number of possible actions
    nF = int(math.pow(gridSize/blockSize, 2))   # Number of features = (12/2)^2 = 36 features

    T = np.zeros((nS, nS, nA))  # Transition function numpy array.
    for y in range(gridSize):
        for x in range(gridSize):
            s = loc2s(x, y, gridSize)   # Assigning a state number to each x,y pair using loc2s
            ns = np.zeros(nA).astype(int)
            ns[0] = loc2s(x, y + 1, gridSize) # N  # Creating a list of possible states you
            ns[1] = loc2s(x + 1, y, gridSize) # E  # could reach by performing the available
            ns[2] = loc2s(x - 1, y, gridSize) # W  # list of actions in each state.
            ns[3] = loc2s(x, y - 1, gridSize) # S
            for a in range(nA):
                for a2 in range(nA):
                    T[ns[a2], s, a] = T[ns[a2], s, a] + (noise / nA); # Adding noise per action to each transition 
                T[ns[a], s, a] = T[ns[a], s, a] + (1 - noise);    # Rest of the prob given to the intended action


    # assign state feature
    # Why are number of features calculated this way?
    # What does each state having that many features intuitively mean?
    F = np.zeros((nS, nF)).astype(int)
    for y in range(gridSize):
        for x in range(gridSize):
            s = loc2s(x, y, gridSize)
            i = math.floor(x / blockSize)
            j = math.floor(y / blockSize)
            f = loc2s(i, j, int(gridSize/blockSize))    
            F[s, f] = 1 # Creating a feature vector for each state.


    start = np.ones((nS, 1))
    # Example: np.ones((2, 1))
    # array([[1.],
    #        [1.]])
    
    start = start / np.sum(start)   # This matrix will contain the value 1/144 in a 144*1 sized matrix
                                    # This gives an equiprobable value for all states to be the start state

    mdp.name = 'gridworld_' + str(gridSize) + 'x' + str(blockSize)
    mdp.gridSize = gridSize
    mdp.blockSize = blockSize
    mdp.nStates = nS
    mdp.nActions = nA
    mdp.nFeatures = nF
    mdp.discount = discount
    mdp.start = start
    mdp.transition = T
    mdp.F = np.tile(F, (nA, 1))
    mdp.weight = None
    mdp.reward = None

    return mdp

def loc2s(x, y, gridSize):  # Location x,y value to state mapping
    x = max(0, min(gridSize - 1, x))
    y = max(0, min(gridSize - 1, y))
    return y * gridSize + x