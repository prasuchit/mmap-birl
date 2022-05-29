import numpy as np
import utils
import utils3
import models
from scipy import full, sparse
import generator
import solver
import copy
import math as m
import time

''' Picked/AtHome - means Sawyer is in hover plane at home position
    Placed - means placed back on conveyor after inspecting and finding status as good
    onionLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    actList = {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion'} '''
''' For pick inspect, correct policy should have:
        140 - 3 # Pick
        143 - 0 # Inspect
        If 111 - 2 or 136 - 3
        101 - 2 # bad onion place in bin
        138 - 4 # Claim after putting in bin
        117 - 1 # good onion place on conv  # This is where the behaviors split off. With pick inspect, roll's 
        128 - 4 # Claim after putting on conv '''
def init(nOnionLoc, nEEFLoc, nPredict, discount, useSparse, noise=0.05):

    nS = nOnionLoc*nEEFLoc*nPredict
    nA = 5
    # 4 combinations of (predicting good/bad and putting it on conveyor/bin) + 1 to stay still
    # + 1 to claim new onion + 1 to create a new list + 1 to pick a good sorting cycle

    nF = 6
    T = np.zeros((nS, nS, nA))    # state transition probability
    F = np.zeros((nS, nF))         # state feature
    start = np.zeros((nS, 1))
    for s in range(nS):
        onionLoc, eefLoc, pred, start = utils3.sid2vals(s, nOnionLoc, nEEFLoc, nPredict, start)
        actidx = utils3.getValidActions(onionLoc, eefLoc, pred)
        f = np.zeros((nF, 1))
        for a in range(nA):
            nextStates = utils3.findNxtStates(onionLoc, eefLoc, pred, a)
            for nxtS in nextStates:
                ns = utils3.vals2sid(nxtS[0], nxtS[1], nxtS[2], nOnionLoc, nEEFLoc, nPredict)
                
                if a not in actidx:     # If action is invalid
                    if not (utils3.isValidNxtState(a, nxtS[0], nxtS[1], nxtS[2])):
                        T[ns, s, a] = 1
                        # T[ns, s, a] = 0.9
                        # for i in range(nS):
                        #     if i != ns:
                        #         T[i, s, a] = 0.1/(nS-1) # Just to prevent any state from having det transitions
                    else:
                        T[43, s, a] = 1   # If next state is valid send it to the sink
                        # T[43, s, a] = 0.9   # If next state is valid send it to the sink
                        # for i in range(nS):
                        #     if i != 91:
                        #         T[i, s, a] = 0.1/(nS-1) # Just to prevent any state from having det transitions
                else:
                    if not (utils3.isValidState(onionLoc, eefLoc, pred)):  # Invalid state
                        if not (utils3.isValidNxtState(a, nxtS[0], nxtS[1], nxtS[2])):
                            T[ns, s, a] = 1
                            # T[ns, s, a] = 0.9
                            # for i in range(nS):
                            #     if i != ns:
                            #         T[i, s, a] = 0.1/(nS-1) # Just to prevent any state from having det transitions
                        else:
                            # Valid actions in invalid states leading to valid next states become a sink
                            T[43, s, a] = 1   # If next state is valid send it to the sink
                            # T[43, s, a] = 0.9   # If next state is valid
                            # for i in range(nS):
                            #     if i != 91:
                            #         T[i, s, a] = 0.1/(nS-1) # Just to prevent any state from having det transitions
                    else:
                        # Valid action in a valid state leading to a valid next state also has a
                        # small failure rate given by noise.
                        if T[s, s, a] == 0:
                            T[s, s, a] += noise # Noise must only be added once
                        # Succeding in intended action with high prob
                        T[ns, s, a] += (1 - noise)/len(nextStates)
                
                # Calculate features
                if pred == 1 and nxtS[0] == 0:
                    f[0] = 1
                    
                if pred == 0 and nxtS[0] == 0:
                    f[1] = 1
                    
                if pred == 1 and nxtS[0] == 2:
                    f[2] = 1
                    
                if pred == 0 and nxtS[0] == 2:
                    f[3] = 1
                    
                # # Stay still
                # if (s == ns):
                #     f[4] = 1

                # Claim new onions
                if pred == 2 and ((onionLoc == 2 or onionLoc == 0) and nxtS[1] == 3):
                    f[4] = 1

                # Pick if unknown
                if onionLoc == 0 and pred == 2 and nxtS[2] == 2 and nxtS[0] == 3:
                    f[5] = 1
 
        F[s, :] = np.transpose(f)

    # Check transition probability
    # for a in range(nA):
    #     for s in range(nS):
    #         err = abs(sum(T[:, s, a]) - 1)
    #         if err > 1e-6 or np.any(T) > 1 or np.any(T) < 0:
    #             print(f"T(:,{s},{a}) = {T[:, s, a]}")
    #             print('ERROR: \n', s, a, np.sum(T[:, s, a]))
    
    assert np.allclose(np.sum(T, axis=0), 1, rtol=1e-5), (
        "un-normalised matrix %s" % T
    )

    # np.savetxt(os.getcwd()+"\csv_files\sorting_T.csv",np.reshape(T,(nS,nS*nA)))
    start = start / np.sum(start)  # Pick inspect
    mdp = models.mdp()
    mdp.name = 'sorting'
    mdp.nStates = nS
    mdp.nActions = nA
    mdp.nFeatures = nF
    mdp.discount = discount
    mdp.useSparse = useSparse
    mdp.start = start
    mdp.F = np.tile(F, (nA, 1))
    mdp.transition = T
    mdp.weight = np.zeros((nF, 1))
    mdp.reward = np.reshape(np.dot(mdp.F, mdp.weight), (nS, nA))
    mdp.sampled = False

    if mdp.useSparse:
        mdp.transitionS = {}
        mdp.rewardS = {}
        mdp.F = sparse.csr_matrix(mdp.F)
        mdp.weight = sparse.csr_matrix(mdp.weight)
        mdp.start = sparse.csr_matrix(mdp.start)

        for a in range(mdp.nActions):
            mdp.transitionS[a] = sparse.csr_matrix(mdp.transition[:, :, a])
            mdp.rewardS[a] = sparse.csr_matrix(mdp.reward[:, a])

    return mdp
