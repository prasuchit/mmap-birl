import numpy as np
import utils
import solver
import math
import copy
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')

def calNegMarginLogPost(w, trajs, mdp, opts):

    originalInfo = utils.getTrajInfo(trajs, mdp)
    occs = originalInfo.occlusions

    llh = 0
    grad1 = 0
    trajsCopy = copy.copy(trajs)

    print("Compute posterior with marginalization...")
    for s in tqdm(range(mdp.nStates)):
        for a in range(mdp.nActions):
            trajsCopy[occs[0,0], occs[0,1], 0] = s
            trajsCopy[occs[0,0], occs[0,1], 1] = a
            trajInfo = utils.getTrajInfo(trajsCopy, mdp)
            mllh, mgrad1 = calLogLLH(w, trajInfo, mdp, opts)
            llh += mllh
            grad1 += mgrad1
    grad1 = (grad1).reshape(36,1)
    prior, grad2 = calLogPrior(w, opts)  
    post = prior + llh
    grad = grad1 + grad2

    return post, grad

def calNegLogPost(w, trajInfo, mdp, opts):
    
    llh, grad1 = calLogLLH(w, trajInfo, mdp, opts)
    prior, grad2 = calLogPrior(w, opts)
    post = prior + llh
    grad = grad1 + grad2

    return post, grad

def calLogPrior(w, opts):
    if opts.priorType == 'Gaussian':
        x = w - opts.mu;
        prior = np.sum(np.matmul(np.transpose(x), x) * -1 / 2 * math.pow(opts.sigma, 2))
        grad = -x / math.pow(opts.sigma, 2)
    else:
        prior = math.log(1)
        grad = np.zeros(w.shape)
        
    return prior, grad

def calLogLLH(w, trajInfo, mdp, opts):
    piL, VL, QL, H = solver.policyIteration(mdp)
    dQ = calGradQ(piL, mdp)

    nF = mdp.nFeatures
    nS = mdp.nStates
    nA = mdp.nActions
    eta = opts.eta

    BQ = eta * QL
    BQSum = np.log(np.sum(np.exp(BQ), axis=1))
    NBQ = BQ.copy()
    for i in range(nA):
        NBQ[:, i] = NBQ[:, i] - BQSum[:]

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        llh += NBQ[s, a]

    pi_sto = np.exp(BQ)
    pi_sto_sum = np.sum(pi_sto, axis=1)
    for i in range(nA):
        pi_sto[:, i] = pi_sto[:, i] / pi_sto_sum[:]

    dlogPi = np.zeros((nF, nS * nA))
    for f in range(nF):
        z = np.reshape(dQ[f, :], (nS, nA))
        for i in range(nA):
            z[:, i] = z[:, i] - np.sum(pi_sto * np.reshape(dQ[f, :], (nS, nA)), axis=1)[:]
        dlogPi[f, :] = np.reshape(z, (1, nS * nA))

    grad = np.zeros(nF)
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        j = a * nS;
        grad = grad + dlogPi[:, j]

    return llh, grad

def calGradQ(piL, mdp):
    nS = mdp.nStates
    nA = mdp.nActions
    Epi = np.zeros((nS, nS * nA)).astype(int)
    idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    for i in range(nS):
        Epi[i, idx[i]] = 1

    dQ = np.linalg.lstsq(np.eye(nS * nA) - np.matmul(mdp.T, Epi), mdp.F)[0]

    return np.transpose(dQ)
