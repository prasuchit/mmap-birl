import numpy as np
import options
import solver
import math
import utils
import gridworld
import highway3
import random
import time
from scipy import sparse
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem, discount=0.99):
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount, problem.useSparse)

    elif problem.name == 'highway':
        mdp = highway3.init(problem.gridSize, problem.nSpeeds, problem.nLanes, problem.discount, 1, problem.useSparse)

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
    if numOccs > 0 and numOccs <= problem.nSteps:
        random.seed(problem.seed)
        for i in range(problem.nTrajs):
            try:
                occlusions = np.zeros(expertData.nSteps)
                # occlusions[random.sample(range(problem.nSteps), numOccs)] = -1
                occlusions[1 + np.arange(int(expertData.nSteps/3))] = -1
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
    if problem.name == 'gridworld':
        if mdp.useSparse:
            policy, value, _, _ = solver.policyIteration(mdp)
        else:
            policy, value, _, _ = solver.piMDPToolbox(mdp)

        optValue = np.dot(np.transpose(mdp.start), value)

        if mdp.useSparse:
            print(' - Optimal value : ', (optValue[0,0]))
        else:
            print(' - Optimal value : %.4f' % (optValue))
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
        
        # seed = problem.seed  
        print('sample trajectory\n')
        trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        
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
            # print(nFeatures[1 + problem.nLanes + i])
            print(nFeatures[problem.nLanes + i])
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

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed=None):

    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)

    np.random.seed(seed)

    for m in range(nTrajs):
        if mdp.useSparse:
            arr = np.array((mdp.start).todense())
            sample = np.random.multinomial(n=1, pvals=np.reshape(arr, arr.size))
        else:
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start, (mdp.nStates)))
        s = np.squeeze(np.where(sample == 1))
        v = 0
        for h in range(nSteps):
            a = np.squeeze(piL[s])
            r = mdp.reward[s, a]
            v = v + r * math.pow(mdp.discount, h)
            trajs[m, h, :] = [s, a]
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.transition[:, s, a], (mdp.nStates)))
            s = np.squeeze(np.where(sample == 1))
        vList[m] = v
    Vmean = np.mean(vList)
    Vvar = np.var(vList)

    return trajs, Vmean, Vvar 