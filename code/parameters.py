import options
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def setIRLParams(alg=None, restart=0, optimizer='Newton-CG', solverMethod= 'scipy', optimMethod = 'nesterovGrad', normMethod = 'softmax', disp=False):   # L-BFGS-B or Newton-CG
    irlOpts = options.irlOptions()
    irlOpts.alg = alg.name
    irlOpts.llhType = alg.llhType 
    irlOpts.priorType = alg.priorType
    irlOpts.restart = restart 
    irlOpts.showMsg = disp
    irlOpts.optimizer = optimizer 
    irlOpts.lb = -1 
    irlOpts.ub = 1 
    irlOpts.solverMethod = solverMethod
    irlOpts.optimMethod = optimMethod
    irlOpts.normMethod = normMethod   
    irlOpts.alpha = 1   # learning rate

    if irlOpts.optimMethod == 'gradAsc':
        irlOpts.decay = .95
        irlOpts.MaxIter = 100
        irlOpts.stepsize = 1/irlOpts.MaxIter

    elif irlOpts.optimMethod == 'nesterovGrad':
        irlOpts.decay = 0.9
        irlOpts.MaxIter = 100
        irlOpts.stepsize = 1/irlOpts.MaxIter

    if irlOpts.priorType == 'Gaussian':
        irlOpts.mu = 0.0
        irlOpts.sigma = 0.5

    if irlOpts.alg == 'MAP_BIRL' or irlOpts.alg == 'MMAP_BIRL':
        if irlOpts.llhType == 'BIRL':
            irlOpts.eta = 3

    return irlOpts

def setProblemParams(name, iters=10, discount=0.99, nTrajs=10, nSteps=100, nOccs = 0, gridSize=12, blockSize=2, nLanes=3, nSpeeds=2, 
                        sorting_behavior = 'pick_inspect', nOnionLoc = 5, nEEFLoc = 4, nPredict = 3, nlistIDStatus = 3, noise = 0.3, 
                        obsv_noise = False, seed=None, useSparse = 0):
    problem = options.problem()
    problem.name = name
    problem.iters = np.arange(iters)
    problem.nExps = len(problem.iters)
    problem.discount = discount
    problem.nExperts = 1
    problem.nTrajs = nTrajs
    problem.nSteps = nSteps
    problem.initSeed = 1
    problem.seed = seed
    problem.nOccs = nOccs
    problem.useSparse = useSparse
    problem.obsv_noise = obsv_noise

    if problem.name == 'gridworld':
        problem.gridSize = gridSize
        problem.blockSize = blockSize
        problem.noise = noise
        problem.filename = name + '_' + str(problem.gridSize) + 'x' + str(problem.blockSize)

    elif problem.name == 'highway':
        problem.nSpeeds  = nSpeeds
        problem.nLanes   = nLanes
        problem.gridSize = gridSize
        problem.filename = name + '_' + str(problem.gridSize) + 'x' + str(problem.nLanes)
    
    elif problem.name == 'sorting':
        problem.sorting_behavior = sorting_behavior
        problem.nOnionLoc = nOnionLoc
        problem.nEEFLoc = nEEFLoc
        problem.nPredict = nPredict
        problem.nlistIDStatus = nlistIDStatus
        problem.noise = noise
        problem.filename = name + '_' + str(problem.nOnionLoc) + 'x' + str(problem.nPredict)
    return problem