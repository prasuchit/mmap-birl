import numpy as np
import options
import solver
import math
import utils
# import scipy.io as sio
import gridworld
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem, discount=0.9):
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount)
    
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.tile(np.eye(nS), (nA, 1)) # Create an identity matrix of size (number of states) in each entry of a nA by 1 matrix
    mdp.T = mdp.transition.copy()
    mdp.T = np.reshape(mdp.T, (nS, nS * nA), order='F') # Flattening the s',s,a matrix to get a 2D matrix of size nS*(nS*nA)
    mdp.T = mdp.discount * np.transpose(mdp.T)  # Discounted transition probabilities
    mdp.E = I - mdp.T   # Eq 5 from Algo for IRL: Vpi = (I - gamma*T)^-1 * R
    return mdp

def generateDemonstration(mdp, problem, numOccs=0):
    data = options.demonstrations()
    nF = mdp.nFeatures
    # Initially mdp.weight = None
    if mdp.weight is None:
        w = utils.sampleWeight(problem.name, nF, problem.seed)  # Returns the weights assigned by expert for the problem
    else:
        w = mdp.weight  # Uses the weight received as input parameter. 
    mdp = utils.convertW2R(w, mdp)  # Returns mdp.reward as a weighted linear combination of features
    trajs, policy = generateTrajectory(mdp, problem)
    data.weight = w
    data.policy = policy
    data.trajId = np.ones(problem.nTrajs).astype(int)
    data.nTrajs = problem.nTrajs
    data.nSteps = problem.nSteps
    stateList = trajs[:,:,0].squeeze()
    actionList = trajs[:,:,1].squeeze()
    trueReward = []
    for s,a in zip(stateList,actionList):
        trajReward = 0
        if (problem.nTrajs > 1):
            for s1,a1 in zip(s,a):
                # print("Reward: ", mdp.reward[s1,a1])
                trajReward += mdp.reward[s1,a1]
        else: trajReward += mdp.reward[s,a]
        trueReward.append(trajReward)

    data.trueReward = max(trueReward)

    if numOccs > 0:
        np.random.seed(problem.seed)
        dist = np.ones(problem.nSteps) / problem.nSteps # Each value in the dist matrix will be probab of (1/nSteps)
        
        # Next line is part of the original code, but not needed because the same thing is done inside the for loop again

        # occlusions = np.random.multinomial(n = numOccs, pvals = dist)   
        
        ###########################################################################################
        # Pick n samples for the n occluded observations from the prob distribution of nsteps
        # Essentially what this is doing is out of the number of trials (given by numOccs here),
        # it is assigning one of the observations to be occluded in each trial and returns a list
        # with the count of the number of times each observation was assigned as occluded in the 
        # n trials we ran.

        # eg: Throw a dice 20 times:
        #   np.random.multinomial(20, [1/6.]*6, size=1)
        # returns array([[4, 1, 7, 5, 2, 1]])
        # It landed 4 times on 1, once on 2, etc.
        ###########################################################################################

        for i in range(problem.nTrajs):
            occlusions = np.random.multinomial(n=numOccs, pvals = dist) # Pick n samples for the n occluded observations from the prob distribution of nsteps
            # So here, in each iteration of the loop, one observation is assigned as occluded with equal prob.
            for j in range(problem.nSteps):
                if occlusions[j] == 1:
                    trajs[i, j, 0] = -1     # The 3rd index holds the s-a pair
                    trajs[i, j, 1] = -1     # 0 - state; 1 - action
                # The value for that traj and that step is assigned -1
    data.trajSet = trajs

    return data

def generateTrajectory(mdp, problem):
    print('Generate Demonstrations')
    nF = mdp.nFeatures
    if mdp.weight is None or mdp.reward is None:
        w = utils.sampleWeight(problem.name, nF, problem.seed)  # Creates a zeros matrix of the size of feature vector.
        print('  - sample a new weight')
        mdp = utils.convertW2R(w, mdp)  # The mdp returned here has a reward vector 
                                        # that is a weighted linear combination of feature values
        print('  - assign weight to the problem')
        print(w)
    policy, value, _, _ = solver.policyIteration(mdp)
    score = np.matmul(np.transpose(mdp.start), value)
    print(' - Total value of this policy: %.4f' % (score))
    trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
    print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
    # print('============ Done =============')

    return trajs, policy

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed=None):

    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)

    np.random.seed(seed)

    for m in range(nTrajs):
        # sample initial state
        sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start, (mdp.nStates))) # This will sample once from all the available start states, pick one at random and assign it a value of 1
        s = np.squeeze(np.where(sample == 1))   # We get back one state value as 1 and we choose that as the start state here
        v = 0
        for h in range(nSteps):
            a = np.squeeze(piL[s])
            r = mdp.reward[s, a]
            v = v + r * math.pow(mdp.discount, h)
            trajs[m, h, :] = [s, a]
            # sample next state
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.transition[:, s, a], (mdp.nStates)))   # Pick the transition prob of any one state
            s = np.squeeze(np.where(sample == 1))

        vList[m] = v
    # Expert's policy value
    Vmean = np.mean(vList)  # Returning the mean and variance of all the value of all the states
    Vvar = np.var(vList)

    return trajs, Vmean, Vvar 