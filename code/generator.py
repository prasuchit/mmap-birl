import numpy as np
import options
import solver
import math
import utils
import gridworld
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem, discount=0.9):
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount)
    
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.tile(np.eye(nS), (nA, 1))
    mdp.T = mdp.transition
    mdp.T = np.reshape(mdp.T, (nS, nS * nA), order='F')
    mdp.T = mdp.discount * np.transpose(mdp.T)
    mdp.E = I - mdp.T 
    return mdp

def generateDemonstration(mdp, problem, numOccs=0):
    data = options.demonstrations()
    nF = mdp.nFeatures
    if mdp.weight is None:
        w = utils.sampleWeight(problem.name, nF, problem.seed)
    else:
        w = mdp.weight
    mdp = utils.convertW2R(w, mdp)
    trajs, policy = generateTrajectory(mdp, problem)
    data.weight = w
    data.policy = policy
    data.trajId = np.ones(problem.nTrajs).astype(int)
    data.nTrajs = problem.nTrajs
    data.nSteps = problem.nSteps

    if numOccs > 0:
        np.random.seed(problem.seed)
        dist = np.ones(problem.nSteps) / problem.nSteps 
        for i in range(problem.nTrajs):
            occlusions = np.random.multinomial(n=numOccs, pvals = dist) 
            for j in range(problem.nSteps):
                if occlusions[j] == 1:
                    trajs[i, j, 0] = -1     # The 3rd index holds the s-a pair
                    trajs[i, j, 1] = -1     # 0 - state; 1 - action
    data.trajSet = trajs

    return data

def generateTrajectory(mdp, problem):
    print('Generate Demonstrations')
    nF = mdp.nFeatures
    if mdp.weight is None or mdp.reward is None:
        w = utils.sampleWeight(problem.name, nF, problem.seed)
        print('  - sample a new weight')
        mdp = utils.convertW2R(w, mdp)
        print('  - assign weight to the problem')
        print(w)
    policy, value, _, _ = solver.policyIteration(mdp)
    optValue = np.matmul(np.transpose(mdp.start), value)
    print(' - Total value of this policy: %.4f' % (optValue))
    trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
    print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))

    return trajs, policy

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed=None):

    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)

    np.random.seed(seed)

    for m in range(nTrajs):
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