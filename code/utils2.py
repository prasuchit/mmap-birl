import utils
import llh
import solver
import numpy as np
import copy
import time
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp

def gradientDescent(mdp, trajs, opts, currWeight = 0, currGrad = 0, cache = []):
    print("======== MAP Inference ========")
    print("======= Gradient Ascent =======")
    for i in range(opts.MaxIter):    # Finding this: R_new = R + δ_t * ∇_R P(R|X)
        print("- %d iter" % (i))
        weightUpdate = (opts.stepsize * opts.alpha * currGrad)
        opts.alpha *= opts.decay
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + weightUpdate
        opti = reuseCacheGrad(currWeight, cache, mdp.useSparse)
        if opti is None:
            print("  No existing cached gradient reusable ")
            pi, H = computeOptmRegn(mdp, currWeight)
            post, currGrad = llh.calcNegMarginalLogPost(currWeight, trajs, mdp, opts)
            cache.append([pi, H, currGrad])
        else:
            print("  Found reusable gradient ")
            currGrad = opti[2]
    return currWeight

def nesterovAccelGrad(mdp, trajs, opts, currWeight = 0, currGrad = 0, cache = []):
    print("======== MAP Inference ========")
    print("==== Nesterov Accel Gradient ====")
    prevGrad = np.copy(currGrad)
    for i in range(opts.MaxIter):    # Finding this: R_new = R + δ_t * ∇_R P(R|X)
        print("- %d iter" % (i))
        # Step 1 - Partial update
        weightUpdate = (opts.decay * prevGrad)
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + (opts.stepsize/2 * weightUpdate)
        opti = reuseCacheGrad(currWeight, cache, mdp.useSparse)
        if opti is None:
            print("  No existing cached gradient reusable ")
            pi, H = computeOptmRegn(mdp, currWeight)
            post, currGrad = llh.calcNegMarginalLogPost(currWeight, trajs, mdp, opts)
            cache.append([pi, H, currGrad])
        else:
            print("  Found reusable gradient ")
            currGrad = opti[2]
        # Step 2 - Full update
        weightUpdate = (opts.decay * prevGrad) + (opts.alpha * currGrad)
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + (opts.stepsize/2 * weightUpdate)
        prevGrad = currGrad
    return currWeight

def processOccl(trajs, nS, nA, nTrajs, nSteps, transition):

    occlusions = []    
    cnt = np.zeros((nS, nA))
    # occlPerTraj = [0]*nTrajs
    for m in range(nTrajs):
        for h in range(nSteps):
            s = trajs[m, h, 0]
            a = trajs[m, h, 1]
            if -1 in trajs[m, h, :]:
               occlusions.append([m,h])
            #    occlPerTraj[m] += 1
            else:
                cnt[s, a] += 1
    '''
    # We use bidirectional search logic to find the reachable states from the state before the occluded step in the traj.
    # Similarly, we can use the state after the occl(s).
    '''
    startPass = time.time()
    ''' Forward Pass '''

    allOccNxtSts = []   # Each sublist within (have next states) corresponds to that index in occlusions.
    for o in occlusions:

        nxtStates = []  # This is to create a list for each occl to hold its possible states.

        if(o[1] - 1 == -1): # If prev index to this occl step is not within 0th step. Meaning that the occl is at 0th step.
            if(o[1] + 1 < nSteps and -1 not in trajs[o[0], o[1] + 1,:]):    # Check if next step is within nSteps and isn't occluded
                for i in range(nS):     # For all current states
                    for a in range(nA): # And all actions
                        if transition[trajs[o[0], o[1] + 1,0], i, a] != 0:  # Which of these current states land me in that next state in the traj?
                            if i not in nxtStates:  # If we don't already have that next state in the list
                                nxtStates.append(i)
                allOccNxtSts.append(nxtStates)
            else:    # If the next step is not within nSteps and/or the next step is also occluded
                allOccNxtSts.append([i for i in range(nS)]) # Current state is all possible states we have.

        elif(o[1] - 1 != -1 and -1 not in trajs[o[0], o[1] - 1,:]): # If occl is not at 0th step and prev step to occl is unoccluded.
            for i in range(nS):     # For all next states
                if transition[i, trajs[o[0], o[1] - 1, 0], trajs[o[0], o[1] - 1, 1]] != 0:  # If transition from that prev state in traj to this next
                                                                                            # state is probable for the action performed there
                    if(o[1] + 1 < nSteps and -1 not in trajs[o[0], o[1] + 1,:]):    # If step after occl is within nSteps and unoccluded
                        for a in range(nA):     # For all possible actions
                            if transition[trajs[o[0], o[1] + 1,0], i, a] != 0:  # Does this potential state behind the occl land me in that next state?
                                if i not in nxtStates:  # If we don't already have that next state in the list
                                    nxtStates.append(i)
                    else:   # If step after occl isn't within nSteps and/or is also occluded
                        if i not in nxtStates:  # If we don't already have that next state in the list
                            nxtStates.append(i)
            allOccNxtSts.append(nxtStates)

        elif(o[1] - 1 != -1 and -1 in trajs[o[0], o[1] - 1,:]): # If occl is not at 0th step and prev step to occl is also occluded.
            for i in range(nS): # For all next states
                for s in allOccNxtSts[occlusions.index(o) -1]:  # From all possible states for prev occluded step
                    for a in range(nA):     # And all actions
                        if transition[i, s, a] != 0:    # If transition to a state is possible, that could be our current occluded state
                            if(o[1] + 1 < nSteps and -1 not in trajs[o[0], o[1] + 1,:]):    # If step after occl is within nSteps and unoccluded
                                for act in range(nA):     # And all actions
                                    if transition[trajs[o[0], o[1] + 1,0], i, act] != 0:  # Does any action from this state land me in that next state?
                                        if i not in nxtStates:  # If we don't already have that next state in the list
                                            nxtStates.append(i)
                                   
                            else:   # If step after occl isn't within nSteps and/or is also occluded
                                if i not in nxtStates:  # If we don't already have that next state in the list
                                    nxtStates.append(i)
            allOccNxtSts.append(nxtStates)
        
        else:    # If none of these conditions match
            allOccNxtSts.append([i for i in range(nS)]) # Current state is all possible states we have.

    ''' Backward pass '''
    for m in range(nTrajs):
        for h in range(nSteps):
            if -1 in trajs[m,h,:] and h != nSteps - 1:
                if -1 not in trajs[m,h+1,:]:
                    tempList = []
                    p = h
                    while(-1 in trajs[m,p,:] and -1 in trajs[m,p-1,:]):
                        occIndex = occlusions.index([m,p])
                        for s in allOccNxtSts[occIndex]:
                            for i in range(nS): # For all next states
                                for a in range(nA):     # And all actions
                                    if transition[s, i, a] != 0:    # If transition to a state is possible, that could be our current occluded state
                                        tempList.append(i)
                        # print(tempList)
                        for k in list(allOccNxtSts[occIndex - 1]):  # Iterating over a copy of the list to allow
                                                                    # modifications to the original list.
                            if(k not in tempList):
                                allOccNxtSts[occIndex - 1].remove(k)
                        p -= 1
            
    endPass = time.time()
    print("Time taken for bidirectional search: ", endPass - startPass)

    return occlusions, cnt , allOccNxtSts
    
def computeOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    if mdp.useSparse:
        piL, _, _, H = solver.policyIteration(mdp)
    else:
        piL, _, _, H = solver.piMDPToolbox(mdp)
    return piL, H

def reuseCacheGrad(w, cache, useSparse):
    for opti in cache:
        if useSparse:
            H = opti[1].todense()
        else:
            H = opti[1]
        constraint = np.dot(H, w)
        compare = np.where(constraint < 0)
        if compare[0].size > 0:
            return opti
    return None


def piInterpretation(policy, name):
    actions = {}
    if name == 'gridworld':
        for i in range(len(policy)):
            if(policy[i] == 0):
                actions[i] = 'North'
            elif(policy[i] == 1):
                actions[i] = 'East'
            elif(policy[i] == 2):
                actions[i] = 'West'
            elif(policy[i] == 3):
                actions[i] = 'South'
    else:
        print("Problem is not gridworld. This function doesn't work for other problems yet.")
    return actions

def computeResults(expertData, mdp, wL):

    mdp = utils.convertW2R(expertData.weight, mdp)
    if mdp.useSparse:
        piE, VE, QE, HE = solver.policyIteration(mdp)
        vE = np.array(np.dot(np.dot(expertData.weight.T,HE.todense().T),mdp.start.todense()))
        QE = np.array(QE.todense())
    else:
        piE, VE, QE, HE = solver.piMDPToolbox(mdp)
        vE = np.matmul(np.matmul(expertData.weight.T,HE.T),mdp.start)

    mdp = utils.convertW2R(wL, mdp)
    if mdp.useSparse:
        piL, VL, QL, HL = solver.policyIteration(mdp)
        vL = np.array(np.dot(np.dot(expertData.weight.T,HL.todense().T),mdp.start.todense()))
        QL = np.array(QL.todense())
    else:
        piL, VL, QL, HL = solver.piMDPToolbox(mdp)
        vL = np.matmul(np.matmul(expertData.weight.T,HL.T),mdp.start)

    d  = np.zeros((mdp.nStates, 1))
    for s in range(mdp.nStates):
        ixE = QE[s, :] == max(QE[s, :])
        ixL = QL[s, :] == max(QL[s, :])
        if ((ixE == ixL).all()):
            d[s] = 0
        else:
            d[s] = 1

    rewardDiff = np.linalg.norm(expertData.weight - wL)
    ''' The value difference compares the visitation freq of expert and 
    learner wrt true weights to find the diff in value they acrue '''
    valueDiff  = abs(vE - vL)   # ILE - Inverse Learning Error
    policyDiff = np.sum(d)/mdp.nStates  # LBA - Learned Behavior Accuracy

    return rewardDiff, valueDiff, policyDiff, piL, piE

def normalizedW(weights, normMethod):
    if normMethod == 'softmax':
        wL = (np.exp(weights))/(np.sum(np.exp(weights))) # Softmax normalization
    elif normMethod == '0-1':
        wL = (weights-min(weights))/(max(weights)-min(weights)) # Normalizing b/w 0-1
    else:   wL = weights # Unnormalized raw weights

    return wL

def logsumexp_row_nonzeros(X):
    result = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        result[i] = logsumexp(X.data[X.indptr[i]:X.indptr[i+1]])
    return result