import options
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def paramsSEIRL(alg=None, restart=0, optimizer='L-BFGS-B', disp=False):
    params = options.irlOptions()
    params.alg = alg.name
    params.llhType = alg.llhType    # Log likelihood type
    params.priorType = alg.priorType
    params.restart = restart  # num of random restart
    params.showMsg = disp
    params.optimizer = optimizer    # L-BFGS-B is Limited memory quasi Newton method 
                                    # for approximating the Broyden–Fletcher–Goldfarb–Shanno algorithm
    params.lb = -1 # lower bounds of reward
    params.ub = 1  # upper bounds of reward

    if params.priorType == 'Gaussian':
        params.mu = 0.0
        params.sigma = 0.1

    if params.alg == 'MAP_BIRL':
        if params.llhType == 'BIRL':
            params.eta = 1.0  # inverse temperature

    return params

def problemParamsSE(name, iters=10, discount=0.9, nTrajs=10, nSteps=100, seed=None):
    problem = options.problem() # Creating class object
    # Setting values for class parameters
    problem.name = name
    problem.iters = np.arange(iters)
    problem.discount = discount
    problem.nExps = len(problem.iters)
    problem.nExperts = 1
    problem.nTrajs = nTrajs
    problem.nSteps = nSteps
    problem.initSeed = 1
    problem.seed = seed

    if problem.name == 'gridworld':
        problem.gridSize = 12
        problem.blockSize = 2
        problem.noise = 0.3
        problem.filename = name + '_' + str(problem.gridSize) + 'x' + str(problem.blockSize)

    return problem
