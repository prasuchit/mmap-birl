import numpy as np
import options
import solver
import math
import utils
import utils3
import gridworld
import highway3
import sorting
import random
import time
from scipy import sparse
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem):
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount, problem.useSparse)

    elif problem.name == 'highway':
        mdp = highway3.init(problem.gridSize, problem.nSpeeds, problem.nLanes, problem.discount, problem.useSparse)

    elif problem.name == 'sorting':
        mdp = sorting.init(problem.nOnionLoc, problem.nEEFLoc, problem.nPredict, problem.nlistIDStatus, problem.sorting_behavior, problem.discount, problem.useSparse, problem.noise)
    
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.tile(np.eye(nS), (nA, 1))
    mdp.T = mdp.transition
    mdp.T = np.reshape(mdp.T, (nS, nS * nA), order='F')
    mdp.T = mdp.discount * np.transpose(mdp.T)
    mdp.E = I - mdp.T
    mdp.nOccs = problem.nOccs
    return mdp

def generateDemonstration(mdp, problem, numOccs=0):
    '''
    @brief  Generates trajectories using simulate technique and
            for each traj, generates a list of random indices of
            len nOccs within range of nSteps where occl are placed.
    '''
    expertData = options.demonstrations()
    nF = mdp.nFeatures
    if mdp.sampled is False:
        w = utils.sampleWeight(problem, nF, problem.seed)
        mdp.sampled = True
    else:
        w = mdp.weight
    mdp = utils.convertW2R(w, mdp)
    trajs, policy = generateTrajectory(mdp, problem)
    expertData.seed = problem.seed
    expertData.weight = w
    expertData.policy = policy
    expertData.trajId = np.ones(problem.nTrajs).astype(int)
    expertData.nTrajs = problem.nTrajs
    expertData.nSteps = problem.nSteps
    if numOccs > 0 and numOccs < problem.nSteps:
        random.seed(problem.seed)
        for i in range(problem.nTrajs):
            try:
                occlusions = np.zeros(expertData.nSteps)
                occlusions[random.sample(range(problem.nSteps), numOccs)] = -1
                # occlusions[1 + np.arange(int(expertData.nSteps/3))] = -1
                # occlusions[(expertData.nSteps - 2) - np.arange(int(expertData.nSteps/3))] = -1
            except ValueError:
                print("ERROR: Number of occlusions exceed total number of steps. Exiting!")
                raise SystemExit(0)
            for j in range(problem.nSteps):
                if occlusions[j] == -1:
                    trajs[i, j, 0] = occlusions[j]     # The 3rd index holds the s-a pair
                    trajs[i, j, 1] = occlusions[j]     # 0 - state 1 - action

    expertData.trajSet = trajs

    return expertData

def generateTrajectory(mdp, problem):
    print('Generate Demonstrations')
    nF = mdp.nFeatures
    if mdp.sampled is False:
        w = utils.sampleWeight(problem, nF, problem.seed)
        mdp.sampled = True
        print('  - sample a new weight')
        mdp = utils.convertW2R(w, mdp)
        print('  - assign weight to the problem')
        print(w)
    if problem.name == 'sorting':
        print(f'solve {mdp.name}\n')
        tic = time.time()  
        if mdp.useSparse:
            policy, value, _, _ = solver.policyIteration(mdp)
        else:
            # policy, value, _, _ = solver.policyIteration(mdp)
            policy, value, _, _ = solver.piMDPToolbox(mdp)

        # np.savetxt("test_expert_policy.csv", policy, delimiter=",")
        toc = time.time()
        elapsedTime = toc - tic
        if mdp.name == 'pick_inspect':
            optValue = np.dot(np.transpose(mdp.start[0]), value)
        else:
            optValue = np.dot(np.transpose(mdp.start[1]), value)
        
        if mdp.useSparse:
            meanThreshold = 1
            varThreshold = 1
            while True:
                trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed, problem.sorting_behavior)
                if abs(abs(optValue[0,0]) - abs(trajVmean)) < meanThreshold and trajVvar < varThreshold:
                    break
        else:
            trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed, problem.sorting_behavior)
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))

    elif problem.name == 'gridworld':
        print(f'solve {mdp.name}\n')
        if mdp.useSparse:
            policy, value, _, _ = solver.policyIteration(mdp)
        else:
            policy, value, _, _ = solver.piMDPToolbox(mdp)

        optValue = np.dot(np.transpose(mdp.start), value)

        if mdp.useSparse:
            print(' - Optimal value : ', (optValue[0,0]))
        else:
            print(' - Optimal value : %.4f' % (optValue))
        
        if mdp.useSparse:
            meanThreshold = 1
            varThreshold = 1
            while True:
                trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
                if abs(optValue[0,0] - trajVmean) < meanThreshold and trajVvar < varThreshold:
                    break
        else:
            trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))

    elif problem.name == 'highway':
        print(f'solve {mdp.name}\n')
        tic = time.time()  
        if mdp.useSparse:
            policy, value, _, _ = solver.policyIteration(mdp)
        else:
            # policy, value, _, _ = solver.policyIteration(mdp)
            policy, value, _, _ = solver.piMDPToolbox(mdp)
        toc = time.time()
        elapsedTime = toc - tic
        
        optValue = np.dot(np.transpose(mdp.start), value)
        
        if mdp.useSparse:
            meanThreshold = 1
            varThreshold = 1
            while True:
                trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
                if abs(optValue[0,0] - trajVmean) < meanThreshold and trajVvar < varThreshold:
                    break
        else:
            trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
        
        nFeatures = np.zeros((nF, 1))
        for t in range(problem.nSteps):
            s = trajs[0, t, 0]
            a = trajs[0, t, 1]
            f = mdp.F[(a)*mdp.nStates + s, :]
            nFeatures += np.reshape(f, (nF,1))

        print('\n# of collisions: ', nFeatures[0])
        print('\n# of lanes     : ')
        for i in range(problem.nLanes):
            print(nFeatures[1 + i])
        print('\n# of speeds    : ')
        for i in range(problem.nSpeeds):
            print(nFeatures[1 + problem.nLanes + i])
        print('\n')
        
        print('\nweight         : ')
        for i in range(mdp.nFeatures):
            print(mdp.weight[i])
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
        optValue = np.dot(np.transpose(mdp.start), value)
        if mdp.useSparse:
            print(' - Optimal value : ', (optValue[0,0]))
        else:
            print(' - Optimal value : %.4f' % (optValue))
        print('Time taken: ', elapsedTime, ' sec\n\n')    
    
    return trajs, policy

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed = None, sorting_behavior = None):

    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)
    if not mdp.useSparse:
        # np.random.seed(seed)
        np.random.seed(None)
    else:
        np.random.seed(None)
    for m in range(nTrajs):
        if mdp.useSparse:
            arr = np.array((mdp.start).todense())
            if mdp.name == 'sorting':
                if sorting_behavior == 'pick_inspect':
                    sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start[0], (mdp.nStates)))
                else:
                    sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start[1], (mdp.nStates)))
            else:
                sample = np.random.multinomial(n=1, pvals=np.reshape(arr, arr.size))
        else:
            if mdp.name == 'sorting':
                if sorting_behavior == 'pick_inspect':
                    sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start[0], (mdp.nStates)))
                else:
                    sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start[1], (mdp.nStates)))
            else:
                # sample = sampleMultinomial(np.reshape(mdp.start, (mdp.nStates)), seed)
                sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start, (mdp.nStates)))
            
        s = np.squeeze(np.where(sample == 1))
        # s = sample
        v = 0
        for h in range(nSteps):
            a = np.squeeze(piL[s])
            r = mdp.reward[s, a]
            v = v + r * math.pow(mdp.discount, h)
            trajs[m, h, :] = [s, a]
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.transition[:, s, a], (mdp.nStates)))
            if mdp.name == 'sorting':
                s = np.squeeze(np.where(sample == 1))
            else:
                s = np.squeeze(np.where(sample == 1))
            # sample = sampleMultinomial(np.reshape(mdp.transition[:, s, a], (mdp.nStates)), seed)
            # s = sample
        vList[m] = v
    Vmean = np.mean(vList)
    Vvar = np.var(vList)
    # trajInfo = utils.getTrajInfo(trajs, mdp)
    return trajs, Vmean, Vvar 


def sampleMultinomial(dist, seed):
    '''
    @brief  Returns the index value of the random sample
            obtained from the distribution provided.
    '''
    np.random.seed(seed)
    x = dist
    s = np.sum(x)
    if s != 1: 
        x = np.divide(x,np.sum(dist))
    sample = (np.argwhere(np.cumsum(x) > np.random.rand(1))[0])
    return sample