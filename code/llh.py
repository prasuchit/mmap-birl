import numpy as np
import utils
import solver
import math
import copy
from tqdm import tqdm
from scipy.special._logsumexp import logsumexp
from multiprocessing import Pool
import time
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

def calcNegMarginalLogPost(w, trajs, mdp, options):

    originalInfo = utils.getOrigTrajInfo(trajs, mdp)
    llh, grad1 = multiProcess(originalInfo, w, trajs, mdp, options)
    # llh, grad1 = serialProcess(originalInfo, w, trajs, mdp, options)
        
    prior, grad2 = calcLogPrior(w, options)
    grad2 = np.reshape(grad2,(mdp.nFeatures,1))
    grad = grad1 + grad2 
    post = prior + llh
    if(options.optiMethod == 'scipy'):
        grad = -np.reshape(grad, mdp.nFeatures)
        post = -post
    elif(options.optiMethod == 'manual'):
        grad = np.reshape(grad, mdp.nFeatures)
        post = post
    else: 
        print("ERROR: Optimization method incorrect")
        raise SystemExit(0)

    if np.isinf(post) or np.isinf(-post) or np.isnan(post):
        print(f'ERROR: prior: %f, llh:%f, eta:%f, w:%f %f \n', prior, llh, options.eta, np.amin(w), np.amax(w)) 
        raise SystemExit(0)
    # print("Posterior inside llh: ", post)
    return post, grad

def multiProcess(originalInfo, w, trajs, mdp, options):

    occs = originalInfo.occlusions
    llh = 0
    grad1 = 0
    mresult = []
    with Pool(processes = 5) as pool:
        if(len(occs) > 0):
            # print("Compute posterior with marginalization...")
            # start_t = time.time()
            for o in tqdm(range(len(occs))):
                trajsCopy = copy.copy(trajs)
                for s in originalInfo.allOccNxtSts[o]:
                    for a in range(mdp.nActions):
                        trajsCopy[occs[o,0], occs[o,1], 0] = s
                        trajsCopy[occs[o,0], occs[o,1], 1] = a
                        trajInfo = utils.getTrajInfo(trajsCopy, mdp)
                        mresult.append(pool.apply_async(calcLogLLH, (w, trajInfo, mdp, options)))

            for i in tqdm(range(len(mresult))):
                mllh, mgrad1 = mresult[i].get()
                llh += mllh
                grad1 += mgrad1
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))
        else:
            # print("No occlusions found...")
            trajsCopy = copy.copy(trajs)
            trajInfo = utils.getTrajInfo(trajsCopy, mdp)
            llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def serialProcess(originalInfo, w, trajs, mdp, options):

    occs = originalInfo.occlusions
    llh = 0
    grad1 = 0
    mresult = []
    if(len(occs) > 0):
        # print("Compute posterior with marginalization...")
        # start_t = time.time()
        for o in tqdm(range(len(occs))):
            trajsCopy = copy.copy(trajs)
            for s in originalInfo.allOccNxtSts[o]:
                for a in range(mdp.nActions):
                    trajsCopy[occs[o,0], occs[o,1], 0] = s
                    trajsCopy[occs[o,0], occs[o,1], 1] = a
                    trajInfo = utils.getTrajInfo(trajsCopy, mdp)
                    mllh, mgrad1 = calcLogLLH(w, trajInfo, mdp, options)
                    llh += mllh
                    grad1 += mgrad1
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))
    else:
        # print("No occlusions found...")
        trajsCopy = copy.copy(trajs)
        trajInfo = utils.getTrajInfo(trajsCopy, mdp)
        llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def calcNegLogPost(w, trajInfo, mdp, options):
    llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
    prior, grad2 = calcLogPrior(w, options)
    grad = grad1 + grad2
    grad = -grad
    post = prior + llh
    post = -post

    if np.isinf(post) or np.isinf(-post):
        print('ERROR: prior: %f, llh: %f, eta: %f, w: %f %f \n', prior, llh, options.eta, np.amin(w), np.amax(w))
    
    return post, grad

def calcLogPrior(w, options):
    if options.priorType == 'Gaussian':
        x = w - options.mu
        prior = np.sum(np.matmul(np.transpose(x), x) * -1 / 2 * math.pow(options.sigma, 2))
        grad = -x / math.pow(options.sigma, 2)
    else:
        prior = math.log(1)
        grad = np.zeros(w.shape)
        
    return prior, grad

def calcLogLLH(w, trajInfo, mdp, options):

    mdp = utils.convertW2R(w, mdp)
    piL, VL, QL, H = solver.piMDPToolbox(mdp) 
    dQ = calcGradQ(piL, mdp)
    nF = mdp.nFeatures
    nS = mdp.nStates
    nA = mdp.nActions
    eta = options.eta
    BQ = eta * QL   
    BQSum = np.reshape(logsumexp(BQ, axis=1),(nS,1))
    NBQ = BQ
    
    NBQ = NBQ - BQSum

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        llh += NBQ[s, a]*n

    # Soft-max policy
    pi_sto = np.exp(NBQ) 

    # calculate dlogPi/dw
    dlogPi = np.zeros((nF, nS * nA))
    z = np.zeros((nS,nA))
    for f in range(nF):
        x = np.reshape(dQ[f, :], (nS, nA), order='F')
        y = np.sum(np.multiply(pi_sto, x), axis=1).reshape(nS,1)
        z = eta * (x - y)  
        dlogPi[f, :] = np.reshape(z, (1, nS * nA), order='F')  

    # Calculating the gradient of the reward function
    grad = np.zeros(nF)
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        j = (a) * nS+s
        grad += n*dlogPi[:, j] 
    return llh, grad

def calcGradQ(piL, mdp):
# Calculate Gradient Q value
    nS = mdp.nStates
    nA = mdp.nActions
    Epi = np.zeros((nS, nS * nA)).astype(int)

    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    dQ = np.linalg.lstsq(np.eye(nS * nA) - np.matmul(mdp.T, Epi), mdp.F)[0]
    return np.transpose(dQ)