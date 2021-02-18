import numpy as np
import utils
import utils2
import utils3
import solver
import math
import copy
from tqdm import tqdm
from scipy.special._logsumexp import logsumexp
from scipy import sparse
from multiprocessing import Pool
import time
# import pymc3 as pm 
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

def calcNegMarginalLogPost(w, trajs, mdp, options):

    # llh, grad1 = multiProcess(w, trajs, mdp, options)
    llh, grad1 = serialProcess(w, trajs, mdp, options)
    prior, grad2 = calcLogPrior(w, options)
    grad2 = np.reshape(grad2,(mdp.nFeatures,1))
    grad = grad1 + grad2 
    post = prior + llh
    if(options.solverMethod == 'scipy'):
        grad = -np.reshape(grad, mdp.nFeatures)
        post = -post
    elif(options.solverMethod == 'manual'):
        grad = np.reshape(grad, mdp.nFeatures)
        post = post
    else: 
        print("ERROR: Optimization method incorrect")
        raise SystemExit(0)

    if np.isinf(post) or np.isinf(-post) or np.isnan(post):
        print(f'ERROR: prior: %f, llh:%f, eta:%f, w:%f %f \n', prior, llh, options.eta, np.amin(w), np.amax(w)) 
        raise SystemExit(0)
    return post, grad

def multiProcess(w, trajs, mdp, options):

    llh = 0
    grad1 = 0
    mresult = []
    with Pool(processes = 5) as pool:
        if(mdp.nOccs > 0):
            originalInfo = utils.getOrigTrajInfo(trajs, mdp)
            occs = originalInfo.occlusions
            # print("Compute posterior with marginalization...")
            # start_t = time.time()
            originalInfo = utils.getOrigTrajInfo(trajs, mdp)
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
            llh, grad1 = calcLogLLH(w, trajInfo, trajs, mdp, options)
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def serialProcess(w, trajs, mdp, options):

    llh = 0
    grad1 = 0
    if(mdp.nOccs > 0):
        originalInfo = utils.getOrigTrajInfo(trajs, mdp)
        occs = originalInfo.occlusions
        # print("Compute posterior with marginalization...")
        # start_t = time.time()
        for o in tqdm(range(len(occs))):
            trajsCopy = copy.copy(trajs)
            for s in originalInfo.allOccNxtSts[o]:
                for a in range(mdp.nActions):
                    trajsCopy[occs[o,0], occs[o,1], 0] = s
                    trajsCopy[occs[o,0], occs[o,1], 1] = a
                    trajInfo = utils.getTrajInfo(trajsCopy, mdp)
                    mllh, mgrad1 = calcLogLLH(w, trajInfo, trajs, mdp, options)
                    llh += mllh
                    grad1 += mgrad1
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))
    else:
        # print("No occlusions found...")
        trajsCopy = copy.copy(trajs)
        trajInfo = utils.getTrajInfo(trajsCopy, mdp)
        llh, grad1 = calcLogLLH(w, trajInfo, trajs, mdp, options)
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def calcNegLogPost(w, trajInfo, trajs, mdp, options):
    llh, grad1 = calcLogLLH(w, trajInfo, trajs, mdp, options)
    prior, grad2 = calcLogPrior(w, options)
    grad = grad1 + grad2
    post = prior + llh
    if(options.solverMethod == 'scipy'):
        grad = -grad
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

def calcLogLLH(w, trajInfo, trajs, mdp, options):

    mdp = utils.convertW2R(w, mdp)
    if mdp.useSparse:
        piL, VL, QL, H = solver.policyIteration(mdp)
    else:
        piL, VL, QL, H = solver.piMDPToolbox(mdp)
        # piL, VL, QL, H = solver.policyIteration(mdp)
    dQ = calcGradQ(piL, mdp)
    nF = mdp.nFeatures
    nS = mdp.nStates
    nA = mdp.nActions
    eta = options.eta
    BQ = eta * QL
    if mdp.useSparse:
        BQSum = np.reshape(utils2.logsumexp_row_nonzeros(BQ),(nS,1))  
    else:
        BQSum = np.reshape(logsumexp(BQ, axis=1),(nS,1))

    NBQ = BQ
    
    NBQ = NBQ - BQSum

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        onionLoc, eefLoc, pred, listIDStatus = utils3.sid2vals(s)
        if listIDStatus == 2:
            method = 0  # 0 - Pick-inspect-place; 1 - Roll-pick-place
        else: 
            method = 1

        if pred != 2:
            obsv_prob = 1 - 0.3*0.95
        else:
            obsv_prob = 1
        # for m in range(trajInfo.nTrajs):
        #     for h in range(trajInfo.nSteps):
        #         if s == trajs[m, h, 0] and a == trajs[m, h, 1]:
        #             if h + 1 != trajInfo.nSteps:
        #                 ns = trajs[m, h+1, 0]
        #                 break
        #     ns = max(mdp.transition[:,s,a])

        llh += np.log(np.nonzero(mdp.start[method])[0][0] + max(mdp.transition[:,s,a]) + obsv_prob)*n*NBQ[s, a]

    # Soft-max policy
    pi_sto = np.exp(NBQ)  # Just pi, not log pi anymore

    # calculate dlogPi/dtheta ; Theta vector is just the weights.
    dlogPi = np.zeros((nF, nS * nA))
    z = np.zeros((nS,nA))
    for f in range(nF):
        x = np.reshape(dQ[f, :], (nS, nA), order='F')
        y = np.sum(np.multiply(pi_sto, x), axis=1).reshape(nS,1)
        z = eta * (x - y)  
        dlogPi[f, :] = np.reshape(z, (1, nS * nA), order='F')  

    # Calculating the gradient of the llh function
    grad = np.zeros(nF)
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        j = (a) * nS + s
        # grad += np.log(np.nonzero(mdp.start[method])[0][0] + max(mdp.transition[:,s,a]) + obsv_prob)*n*(dlogPi[:, j])
        grad += n*(dlogPi[:, j])
    return llh, grad

def calcGradQ(piL, mdp):
# Calculate Gradient Q value
    nS = mdp.nStates
    nA = mdp.nActions
    Epi = np.zeros((nS, nS * nA)).astype(int)

    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    for s in range(nS):
        Epi[s, act_ind[s]] = 1

    if mdp.useSparse:
        Epi = sparse.csr_matrix(Epi)
        dQ = np.linalg.lstsq(sparse.eye(nS * nA) - (mdp.T * Epi), (mdp.F).todense())[0]
    else:
        dQ = np.linalg.lstsq(np.eye(nS * nA) - np.matmul(mdp.T, Epi), mdp.F)[0]
    return np.transpose(dQ)