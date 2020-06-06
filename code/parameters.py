import options
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def setIRLParams(alg=None, restart=0, optimizer='Newton-CG', optiMethod= 'scipy', disp=False):   # L-BFGS-B or Newton-CG
    irlOpts = options.irlOptions()
    irlOpts.alg = alg.name
    irlOpts.llhType = alg.llhType 
    irlOpts.priorType = alg.priorType
    irlOpts.restart = restart 
    irlOpts.showMsg = disp
    irlOpts.optimizer = optimizer 
    irlOpts.lb = -1 
    irlOpts.ub = 1 
    irlOpts.optiMethod = optiMethod
    if irlOpts.priorType == 'Gaussian':
        irlOpts.mu = 0.0
        irlOpts.sigma = 0.1

    if irlOpts.alg == 'MAP_BIRL':
        if irlOpts.llhType == 'BIRL':
            irlOpts.eta = 2.0

    return irlOpts

def setProblemParams(name, iters=10, discount=0.99, nTrajs=10, nSteps=100, gridSize=12, blockSize=2, noise=0.3, seed=None):
    problem = options.problem()
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
        problem.gridSize = gridSize
        problem.blockSize = blockSize
        problem.noise = noise
        problem.filename = name + '_' + str(problem.gridSize) + 'x' + str(problem.blockSize)

    return problem