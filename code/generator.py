import numpy as np
import options
import solver
import math
import utils
import utils3
import gridworld
import highway3
import sorting
import yaml_utils as yu
import random
import time
import copy
from scipy import sparse
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem):
    '''
    @brief Generate a fully defined MDP for a given problem.
    '''
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount, problem.useSparse)

    elif problem.name == 'highway':
        mdp = highway3.init(problem.gridSize, problem.nSpeeds, problem.nLanes, problem.discount, problem.useSparse)

    elif problem.name == 'sorting':
        mdp = sorting.init(problem.nOnionLoc, problem.nEEFLoc, problem.nPredict, problem.discount, problem.useSparse, problem.noise)
    
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.tile(np.eye(nS), (nA, 1))
    mdp.T = mdp.transition
    mdp.T = np.reshape(mdp.T, (nS, nS * nA), order='F')
    mdp.T = mdp.discount * np.transpose(mdp.T)
    mdp.E = I - mdp.T
    mdp.nOccs = problem.nOccs
    return mdp

def generateDemonstration(mdp, problem):
    '''
    @brief  Generates trajectories using simulate technique and
            for each traj, generates a list of random indices of
            len nOccs within range of nSteps where occl are placed.
    '''
    expertData = options.demonstrations()
    numOccs = problem.nOccs
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
                    trajs[i, j, 0] = occlusions[j]     # The 3rd index of trajs holds the s-a pair
                    trajs[i, j, 1] = occlusions[j]     # 0 - state 1 - action

    expertData.trajSet = trajs
    """ TBD: Move this to a seperate function that can be called from runner
    if the domain is passed as a YAML file."""
    # yu.YAMLGenerator(mdp, expertData).writeVals()
    # yu.YAMLGenerator().readVals()
    # expertData.trajSet = np.loadtxt('csv_files/Ehsan_trajs/gated_traj_2.csv', delimiter=',', dtype=int)[np.newaxis,:]
    return expertData

def generateTrajectory(mdp, problem):
    '''
    @brief  Extension of the generateDemonstration function.
    '''
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

        toc = time.time()
        elapsedTime = toc - tic
        optValue = np.dot(np.transpose(mdp.start), value)
        
        if not problem.obsv_noise:
            if mdp.useSparse:
                meanThreshold = 1
                varThreshold = 1
                while True:
                    trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
                    if abs(abs(optValue[0,0]) - abs(trajVmean)) < meanThreshold and trajVvar < varThreshold:
                        break
            else:
                trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
            print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
        else:
            obsvs = utils3.applyObsvProb(problem, policy, mdp, sanet_traj = True)
            return obsvs, policy

    elif problem.name == 'gridworld':
        print(f'solve {mdp.name}\n')
        if mdp.useSparse:
            policy, value, _, _ = solver.policyIteration(mdp)
        else:
            policy, value, _, _ = solver.piMDPToolbox(mdp)

        optValue = np.dot(np.transpose(mdp.start), value)

        if not problem.obsv_noise:
            if mdp.useSparse:
                meanThreshold = 1
                varThreshold = 1
                while True:
                    trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
                    if abs(abs(optValue[0,0]) - abs(trajVmean)) < meanThreshold and trajVvar < varThreshold:
                        break
            else:
                trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
            print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
        else:
            obsvs = utils3.applyObsvProb(problem, policy, mdp)
            return obsvs, policy

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

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed = None):
    '''
    @brief  Extension of generateTrajectories function.
    '''
    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)
    if not mdp.useSparse:
        # np.random.seed(seed)
        np.random.seed(None)
    else:
        np.random.seed(seed)
        # np.random.seed(None)
    for m in range(nTrajs):
        if mdp.useSparse:
            start = np.array((mdp.start).todense())
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start, start.size))
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
            s = np.squeeze(np.where(sample == 1))
            # sample = sampleMultinomial(np.reshape(mdp.transition[:, s, a], (mdp.nStates)), seed)
            # s = sample
        vList[m] = v
    Vmean = np.mean(vList)
    Vvar = np.var(vList)
    # trajInfo = utils.getTrajInfo(trajs, mdp)
    return trajs, Vmean, Vvar

def sampleTauTrajectories(mdp, traj, nTrajs, nSteps, seed = None):
    '''
    @brief Since there could be a gigantic number of trajectories possible
    from a given set of observations depending on the observation model and
    the transition function of the MDP, we sample a large number of trajectories
    here as an approximation. 
    NOTE: Currently, this has only been implemented for the Forestworld and Sorting
    domain and not in the best way. Feel free to improve upon the implementation or
    extend it to your own domain.
    '''
    tautrajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    np.random.seed(seed)
    nS = mdp.nStates
    for m in range(nTrajs):
        for h in range(nSteps):
            if h > 0:
                ps = copy.copy(int(traj[h-1,0]))
            s = copy.copy(int(traj[h,0]))
            a = copy.copy(int(traj[h,1]))
            if s != -1:
                if mdp.name =='sorting':
                    onionLoc, eefLoc, pred = utils3.sid2vals(s)
                    if pred != 2:
                        pred = np.random.choice([pred, int(not pred)], 1)[0]
                    tautrajs[m,h,0] = utils3.vals2sid(onionLoc, eefLoc, pred)
                    tautrajs[m,h,1] = a

                elif mdp.name == 'gridworld':
                    if h > 0 and s == 14 and ps != 13:
                        tautrajs[m,h,0] = np.random.choice([s, 15], 1)[0]
                        tautrajs[m,h,1] = a 
                    else:
                        tautrajs[m,h,0] = s
                        tautrajs[m,h,1] = a
            else:
                tautrajs[m,h,0] = s
                tautrajs[m,h,1] = a
    return tautrajs

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