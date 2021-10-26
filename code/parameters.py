import options
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def setIRLParams(data_loaded):   # L-BFGS-B or Newton-CG
    irlOpts = options.irlOptions()
    irlOpts.alg = data_loaded['algoName']
    irlOpts.llhType = data_loaded['llhName']
    irlOpts.priorType = data_loaded['priorName']
    irlOpts.restart = int(data_loaded['restart'])
    irlOpts.showMsg = data_loaded['disp']
    irlOpts.optimizer = data_loaded['optimizer']
    irlOpts.lb = -1 
    irlOpts.ub = 1 
    irlOpts.solverMethod = data_loaded['solverMethod']
    irlOpts.optimMethod = data_loaded['optimMethod']
    irlOpts.normMethod = data_loaded['normMethod']
    irlOpts.alpha = float(data_loaded['alpha'])   # learning rate

    if irlOpts.optimMethod == 'gradAsc':
        irlOpts.decay = .95
        irlOpts.MaxIter = int(data_loaded['iters'])
        irlOpts.stepsize = 1/irlOpts.MaxIter

    elif irlOpts.optimMethod == 'nesterovGrad':
        irlOpts.decay = 0.9
        irlOpts.MaxIter = int(data_loaded['iters'])
        irlOpts.stepsize = 1/irlOpts.MaxIter

    if irlOpts.priorType == 'Gaussian':
        irlOpts.mu = float(data_loaded['prior_mean'])   
        irlOpts.sigmasq = float(data_loaded['prior_var'])

    if irlOpts.alg == 'MAP_BIRL' or irlOpts.alg == 'MMAP_BIRL':
        if irlOpts.llhType == 'BIRL':
            irlOpts.eta = float(data_loaded['boltz_beta'])

    return irlOpts

def setProblemParams(data_loaded):

    problem = options.problem()
    problem.name = data_loaded['probName']
    problem.iters = np.arange(int(data_loaded['iters']))
    problem.discount = float(data_loaded['discount'])
    problem.nExperts = 1
    problem.nTrajs = int(data_loaded['nTrajs'])
    problem.nSteps = int(data_loaded['nSteps'])
    problem.initSeed = 1
    if data_loaded['seed'] == 'None':
        problem.seed = None
    else: problem.seed = int(data_loaded['seed'])
    problem.nOccs = int(data_loaded['nOccs'])
    problem.useSparse = bool(data_loaded['useSparse'])
    problem.obsv_noise = float(data_loaded['obsv_noise'])

    if problem.name == 'gridworld':
        problem.gridSize = int(data_loaded['init_gridSize'])
        problem.blockSize = int(data_loaded['init_blockSize'])
        problem.noise = float(data_loaded['init_noise'])
        problem.filename = problem.name + '_' + str(problem.gridSize) + 'x' + str(problem.blockSize)

    elif problem.name == 'highway':
        problem.nSpeeds  = int(data_loaded['init_nSpeeds'])
        problem.nLanes   = int(data_loaded['init_nLanes'])
        problem.gridSize = int(data_loaded['init_gridSize'])
        problem.noise = float(data_loaded['init_noise'])
        problem.filename = problem.name + '_' + str(problem.gridSize) + 'x' + str(problem.nLanes)
    
    elif problem.name == 'sorting':
        problem.nOnionLoc = int(data_loaded['nOnionLoc'])
        problem.nEEFLoc = int(data_loaded['nEEFLoc'])
        problem.nPredict = int(data_loaded['nPredict'])
        problem.noise = float(data_loaded['init_noise'])
        problem.trajType = data_loaded['trajType']
        problem.filename = problem.name + '_' + str(problem.nOnionLoc) + 'x' + str(problem.nPredict)
    return problem