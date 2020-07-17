# Generate highway problem [Abbeel & Ng, ICML 2004]
# speed: 2~4
# lanes: 3
import numpy as np
import utils
import models
# from scipy import full, sparse
import generator
import solver
import copy
import math as m
import time
import scipy.io as sio

def init(nGrids, nSpeeds, nLanes, discount, bprint):

    appearanceProb = np.array(np.linspace(0.1,1,num=nLanes, endpoint=False))  # prob. of other car appearing on each lane
    succProb = np.reshape(np.array([0.8, 0.4, 1.0, 0.8]), (2,2))    # prob of successfully moving in intended way
    carSize  = 2
    nS       = int(nSpeeds*nLanes*m.pow(nGrids,nLanes))

    nA = 5                   # nop, move left, move right, speed up, speed down
    nF = (1 + nLanes + nSpeeds)    # collision, lanes, speeds
    T  = np.zeros((nS, nS, nA))    # state transition probability
    F  = np.zeros((nS, nF))         # state feature

    for s  in range(nS):
        # Y gives the locations of the other cars
        spd, myx, Y = utils.sid2info(s, nSpeeds, nLanes, nGrids)
        
        nX = np.zeros((nA, 2))
        nX[0, :] = [spd, myx]                     # nop
        nX[1, :] = [spd, max(0, myx - 1)]         # move left 
        nX[2, :] = [spd, min(nLanes - 1, myx + 1)]    # move right
        nX[3, :] = [min(nSpeeds - 1, spd + 1), myx]   # speed up
        nX[4, :] = [max(0, spd - 1), myx]         # speed down
        idx1 = utils.find(Y, lambda y: y > 0)
        idx2 = utils.find(Y, lambda y: y == 0)

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
            nY[:,idx1[0]] += (spd + 1)
        nY[nY > nGrids - 1] = 0
        nY = np.concatenate((nY,np.zeros((np.shape(nY)[0], 1))), axis=1)

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
            # calculate transition probability
            for i in range(len(nY)):
                ns = int(utils.info2sid(nX[a, 0], nX[a, 1], nY[i, :], nSpeeds, nLanes, nGrids))
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

        # calculate feature
        f = np.zeros((nF, 1))
        
        # check collision
        if Y[myx] > ((nGrids -1) - carSize*2) and Y[myx] < (nGrids-1):
            f[0] = 1
        f[1 + myx] = 1             # lane
        f[1 + nLanes + spd] = 1    # speed
        F[s, :] = np.transpose(f)

    # check transition probability
    for a in range(nA):
        for s in range(nS):
            err = abs(sum(T[:, s, a]) - 1)
            if err > 1e-6 or np.any(T) > 1 or np.any(T) < 0:
                print('ERROR: %d %d %f\n', s, a, np.sum(T[:, s, a]))

    # initial state distribution
    start     = np.zeros((nS, 1))
    s0        = utils.info2sid(0, 1,[0]*nLanes, nSpeeds, nLanes, nGrids)
    start[s0] = 1

    # # weight for reward
    # w = np.zeros((nF, 1))
    # # fast driver avoids collisions and prefers high speed
    # w[1] = -1      # collision
    # w[-1] = 0.1   # high speed

    # # safe driver avoids collisions and prefers right-most lane
    # w(1) = -1
    # w(1 + nLanes) = 0.1

    # # demolition prefers collisions and high-speed
    # w(1) = 1
    # w(end) = 0.1

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
    mdp.useSparse  = 1
    mdp.start      = start
    mdp.F          = np.tile(F, (nA, 1))
    mdp.transition = T
    mdp.weight = None
    mdp.reward = None

    # if mdp.useSparse:
    #     mdp.F      = sparse.csr_matrix(mdp.F)
    #     mdp.weight = sparse.csr_matrix(mdp.weight)
    #     mdp.start  = sparse.csr_matrix(mdp.start)
        
    #     for a in range(mdp.nActions):
    #         mdp.transitionS[a] = sparse.csr_matrix(mdp.transition[:, :, a])
    #         mdp.rewardS[a] = sparse.csr_matrix(mdp.reward[:, a])

    # print("Leaving Highway init")
    
    return mdp