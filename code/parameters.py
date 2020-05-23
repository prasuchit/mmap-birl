import options
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def setIRLParams(alg=None, restart=0, optimizer='L-BFGS-B', disp=False):
    irlOpts = options.irlOptions()
    irlOpts.alg = alg.name
    irlOpts.llhType = alg.llhType    # Log likelihood type
    irlOpts.priorType = alg.priorType
    irlOpts.restart = restart  # num of random restart
    irlOpts.showMsg = disp
    irlOpts.optimizer = optimizer    # L-BFGS-B is Limited memory quasi Newton method 
                                    # for approximating the Broyden–Fletcher–Goldfarb–Shanno algorithm
    irlOpts.lb = -1 # lower bounds of reward
    irlOpts.ub = 1  # upper bounds of reward

    if irlOpts.priorType == 'Gaussian':
        irlOpts.mu = 0.0
        irlOpts.sigma = 0.1

    if irlOpts.alg == 'MAP_BIRL':
        if irlOpts.llhType == 'BIRL':
            irlOpts.eta = 1.0  # inverse temperature

    return irlOpts

def setProblemParams(name, iters=10, discount=0.99, nTrajs=10, nSteps=100, gridSize=12, blockSize=2, noise=0.3, seed=None):
    problem = options.problem() # Creating class object
    # Setting values for class attributes
    problem.name = name
    problem.iters = np.arange(iters)
    problem.discount = discount
    problem.nExps = len(problem.iters)  # Number of experiments?
    problem.nExperts = 1
    problem.nTrajs = nTrajs
    problem.nSteps = nSteps
    problem.initSeed = 1
    problem.seed = seed

    if problem.name == 'gridworld':
        problem.gridSize = gridSize
        problem.blockSize = blockSize
        problem.noise = noise
        problem.filename = name + '_' + str(problem.gridSize) + 'x' + str(problem.blockSize)

    return problem