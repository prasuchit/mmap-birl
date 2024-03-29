# Generate highway problem [Abbeel & Ng, ICML 2004]
# speed: 2~4
# lanes: 3
import numpy as np
import utils
import models
from scipy import full, sparse
import generator
import solver
import copy
import math as m
import time

def init(nGrids, nSpeeds, nLanes, discount, useSparse):

    appearanceProb = np.array(np.linspace(0.4,1,num=nLanes, endpoint=False))  # prob. of other car appearing on each lane
    # prob of successfully moving in intended way
                                  # spd0 spd1
    succProb = np.reshape(np.array([0.8, 0.4,           # Action (1,2), (1,2) 
                                    1.0, 0.8]), (2,2))  # Action (3,4), (3,4)   
    carSize  = 2
    nS       = int(nSpeeds*nLanes*m.pow(nGrids,nLanes))

    nA = 5                   # nop, move left, move right, speed up, speed down
    nF = (1 + nLanes + nSpeeds)    # collision, lanes, speeds
    # nF = (1 + nLanes + 1)    # collision, lanes, high speed
    T  = np.zeros((nS, nS, nA))    # state transition probability
    F  = np.zeros((nS, nF))         # state feature

    for s  in range(nS):
        # Y gives the locations of the other cars
        spd, myx, Y = utils.sid2info(s, nSpeeds, nLanes, nGrids)
        
        nX = np.zeros((nA, 2))
        nX[0, :] = [spd, myx]                         # nop
        nX[1, :] = [spd, max(0, myx - 1)]             # move left 
        nX[2, :] = [spd, min(nLanes - 1, myx + 1)]    # move right
        nX[3, :] = [min(nSpeeds - 1, spd + 1), myx]   # speed up
        nX[4, :] = [max(0, spd - 1), myx]             # speed down
        idx1 = utils.find(Y, lambda y: y > 0)
        idx2 = utils.find(Y, lambda y: y == 0)
        '''
        Y = 1 is a car being in a lane(Given by idx1). 
        Y = 0 is no car in that lane(Given by idx2).
        nY keeps a list of y positions where the next car can appear in that lane. So if idx2 is not
        none, then there's no car in that lane. Therefore, the next Y position can have a car. 
        Index starts from 0, so we add 1 for car to appear at the next y block.
        If idx1 is not none, then, there's already a car there. So next car must appear at a y
        proportional to the speed I'm moving at currently. So that is given by spd+1.
        If that y pose is greater than gridsize, we reset it to zero. Now we concatenate a column 
        of zeros beside this new nY matrix.
        '''
        nY = []
        if idx2 is not None:
            for i in idx2:
                Y2    = copy.copy(Y)
                Y2[i] = Y2[i] + 1
                nY.append(Y2)
        Y2 = copy.copy(Y)
        nY.append(Y2)
        nY = np.array(nY)
        if idx1 is not None:
            nY[:,idx1] += (spd + 1)
        nY[nY > nGrids - 1] = 0
        nY = np.concatenate((nY, np.zeros((np.shape(nY)[0], 1))), axis=1)
        '''
        If there wasn't a car there previously and newY matrix has a car there,
        then, prob of new car appearing is given by appearanceProb based on 
        lane number.
        If there was a car and nY also has a car, then 1-appearanceProb gives the 
        probability of that happening.
        '''
        for i  in range(len(nY)):
            p = 1
            for j in range(nLanes):
                if Y[j] == 0 and nY[i, j] == 1:
                    p = p*appearanceProb[j]
                elif Y[j] == 0 and nY[i, j] == 0:
                    p = p*(1 - appearanceProb[j])
            nY[i, nLanes] = p

        p = 1 - np.sum(nY[:, nLanes])
        nY[-1, nLanes] = nY[-1, nLanes] + p
        for a in range(nA):
            # Calculate transition probability
            for i in range(np.shape(nY)[0]):
                ns = int(utils.info2sid(nX[a, 0], nX[a, 1], nY[i, :], nSpeeds, nLanes, nGrids))
                '''
                succProb is an array with the number of columns = number of speeds
                Number of rows = num_actions/2 because left and right have same prob and so on.
                For each action, the transition noise is a factor of the speed. Lower the speed,
                higher the prob of succeeding. 

                Now, the transition probability is multiplied to appearance probability to
                get the probability of reaching that next state by doing that action.
                And since a wrong action would have you successfuly move into a state where 
                nY has a car, this would cause a collision and can be captured in the features.
                '''
                if a == 1 or a == 2:
                    pr = (np.power(succProb[0, spd], spd))  
                    T[ns, s, a] = T[ns, s, a] + nY[i, -1] * pr
                    ns2 = int(utils.info2sid(spd, myx, nY[i, :], nSpeeds, nLanes, nGrids))
                    T[ns2, s, a] = T[ns2, s, a] + nY[i, -1] * (1.0 - pr)
                elif a == 3 or a == 4:
                    pr = (np.power(succProb[1, spd], spd))
                    T[ns, s, a] = T[ns, s, a] + nY[i, -1] * pr
                    ns2 = int(utils.info2sid(spd, myx, nY[i, :], nSpeeds, nLanes, nGrids))
                    T[ns2, s, a] = T[ns2, s, a] + nY[i, -1] * (1.0 - pr)
                else:
                    T[ns, s, a] = nY[i, -1]

        # Calculate feature
        f = np.zeros((nF, 1))
        
        f[0] = int(Y[myx] > ((nGrids -1) - carSize*2) and Y[myx] < (nGrids-1))  # Check collision
        f[1 + myx] = 1             # lane
        f[1 + nLanes + spd] = 1    # speed
        # f[nLanes + 1] = 1    # speed
        F[s, :] = np.transpose(f)

    # Check transition probability
    for a in range(nA):
        for s in range(nS):
            err = abs(sum(T[:, s, a]) - 1)
            if err > 1e-6 or np.any(T) > 1 or np.any(T) < 0:
                print('ERROR: \n', s, a, np.sum(T[:, s, a]))

    # Initial state distribution
    start     = np.zeros((nS, 1))
    s0        = utils.info2sid(0, 1,[0]*nLanes, nSpeeds, nLanes, nGrids)
    start[s0] = 1
    # generate MDP
    mdp = models.mdp()
    mdp.name       = 'highway'
    mdp.nSpeeds    = nSpeeds
    mdp.nLanes     = nLanes
    mdp.nGrids     = nGrids
    mdp.carSize    = carSize
    mdp.appearProb = appearanceProb
    mdp.succProb   = succProb
    mdp.nStates    = nS
    mdp.nActions   = nA
    mdp.nFeatures  = nF
    mdp.discount   = discount
    mdp.useSparse  = useSparse
    mdp.start      = start
    mdp.F          = np.tile(F, (nA, 1))
    mdp.transition = T
    mdp.weight = np.zeros((nF,1))
    mdp.reward = np.reshape(np.dot(mdp.F,mdp.weight), (nS, nA))
    mdp.useSparse = useSparse
    mdp.sampled = False

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