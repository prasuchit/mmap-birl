import numpy as np
import utils
import utils2
import utils3
import solver
import generator
import math
import copy
from tqdm import tqdm
from scipy.special._logsumexp import logsumexp
from scipy import sparse
from multiprocessing import Pool
import time
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

def calcNegMarginalLogPost(w, trajs, mdp, options, problem):

    # mdp.nOccs = 0

    if not problem.obsv_noise:
        # llh, grad1 = parallelProcess(w, trajs, mdp, options)
        llh, grad1 = serialProcess(w, trajs, mdp, options)
    else:
        # llh, grad1 = parallelProcess_obsv(w, trajs, mdp, options)    # This hasn't been built yet.
        llh, grad1 = serialProcess_obsv(w, trajs, mdp, options)
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
        print(f'ERROR: prior: {prior}, llh: {llh}, eta: {options.eta}, w:{np.amin(w)} {np.amax(w)} \n') 
        raise SystemExit(0)
    return post, grad

def parallelProcess(w, trajs, mdp, options):

    llh = 0
    grad1 = np.zeros(mdp.nFeatures)
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
            llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def serialProcess(w, trajs, mdp, options):

    llh = 0
    grad1 = np.zeros(mdp.nFeatures)
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


def parallelProcess_obsv(w, obsvs, mdp, options):

    llh = 0
    grad1 = np.zeros(mdp.nFeatures)
    mresult = []
    obsvsCopy = copy.copy(obsvs)
    obs_prob = utils3.getObsvInfo(obsvsCopy, mdp)

    with Pool(processes = 5) as pool:
        if(mdp.nOccs > 0):
            originalInfo = utils.getOrigTrajInfo(obsvsCopy, mdp)
            occs = originalInfo.occlusions
            # print("Compute posterior with marginalization...")
            # start_t = time.time()
            for o in tqdm(range(len(occs))):
                for s in originalInfo.allOccNxtSts[o]:
                    for a in range(mdp.nActions):
                        obsvsCopy[occs[o,0], occs[o,1], 0] = s
                        obsvsCopy[occs[o,0], occs[o,1], 1] = a
                        mresult.append(pool.apply_async(calcLogLLH_obsv, (w, obsvsCopy, obs_prob, mdp, options)))

            for i in tqdm(range(len(mresult))):
                mllh, mgrad1 = mresult[i].get()
                llh += mllh
                grad1 += mgrad1
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))
        else:
            print("No occlusions found...")
            llh, grad1 = calcLogLLH_obsv(w, obsvsCopy, obs_prob, mdp, options)
            grad1 = np.reshape(grad1,(mdp.nFeatures,1))

        return llh, grad1

def serialProcess_obsv(w, obsvs, mdp, options):

    llh = 0
    grad1 = np.zeros(mdp.nFeatures)
    obsvsCopy = copy.copy(obsvs)
    obs_prob = utils3.getObsvInfo(obsvsCopy, mdp)

    if(mdp.nOccs > 0):
        originalInfo = utils.getOrigTrajInfo(obsvsCopy, mdp)
        occs = originalInfo.occlusions
        # print("Compute posterior with marginalization...")
        # start_t = time.time()
        for o in tqdm(range(len(occs))):
            for s in originalInfo.allOccNxtSts[o]:
                for a in range(mdp.nActions):
                    obsvsCopy[occs[o,0], occs[o,1], 0] = s
                    obsvsCopy[occs[o,0], occs[o,1], 1] = a
                    mllh, mgrad1 = calcLogLLH_obsv(w, obsvsCopy, obs_prob, mdp, options)
                    llh += mllh
                    grad1 += mgrad1
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))
    else:
        print("No occlusions found...")
        llh, grad1 = calcLogLLH_obsv(w, obsvsCopy, obs_prob, mdp, options)
        grad1 = np.reshape(grad1,(mdp.nFeatures,1))

    return llh, grad1

def calcNegLogPost(w, trajInfo, mdp, options):
    llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
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

def calcLogLLH_obsv(w, obsvs, obs_prob, mdp, options):

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
    nTraj = np.shape(obsvs)[0]
    nSteps = np.shape(obsvs)[1]
    BQ = eta * QL
    if mdp.useSparse:
        BQSum = np.reshape(utils2.logsumexp_row_nonzeros(BQ),(nS,1))  
    else:
        BQSum = np.reshape(logsumexp(BQ, axis=1),(nS,1))

    NBQ = BQ
    
    NBQ = NBQ - BQSum

    # Soft-max policy
    pi_sto = np.exp(NBQ)  # Just pi, not log pi anymore

    if mdp.name == 'sorting':
        sampling_quantity = 100
        if mdp.sorting_behavior == 'pick_inspect':
            start_prob = np.max(mdp.start[0])
        else: start_prob = np.max(mdp.start[1])
    else: 
        sampling_quantity = 100
        start_prob = np.max(mdp.start)

    llh = 0
    grad = np.zeros(nF) # Calculating the gradient of the llh function
    dh_theta_sum = np.zeros(nF)
    for t in range(nTraj):
        t_llh = 0
        t_grad = np.zeros(nF) 
        h_theta_sum = 0
        tau_sa = generator.sampleTauTrajectories(mdp, obsvs[t], sampling_quantity, np.shape(obsvs)[1], None)
        for m in range(sampling_quantity):
            obs_prob_prod = 1
            h_theta = 0
            for h in range(nSteps):
                s = tau_sa[m,h,0]
                a = tau_sa[m,h,1]
                if obs_prob[t,h,s,a] > 0:
                    obs_prob_prod *= obs_prob[t,h,s,a]
                else:
                    obs_prob_prod *= obs_prob[t,h,s,a]
                    break
            
            trans_prob_prod = 1
            pi_sto_prod = 1
            if obs_prob_prod > 0:
                for i in range(1,nSteps):
                    ns = tau_sa[m,i,0]
                    na = tau_sa[m,i,1]
                    s = tau_sa[m,i-1,0]
                    a = tau_sa[m,i-1,1]
                    trans_prob_prod *= mdp.transition[ns,s,a]
                    pi_sto_prod *= pi_sto[ns,na]

                if trans_prob_prod > 0:
                    c_tau = start_prob*obs_prob_prod*trans_prob_prod
                    h_theta = c_tau*pi_sto_prod
                    h_theta_sum += h_theta
                    if h_theta != 0:
                        t_llh += np.log(h_theta) 
                    dh_theta_sum += c_tau*(calc_pi_sto_grad(tau_sa[m], pi_sto, nF, nA, dQ))
                    t_grad = (1/h_theta_sum) * dh_theta_sum
        llh += t_llh
        grad += t_grad
    return llh, grad

def calc_pi_sto_grad(tau_sa, pi_sto, nF, nA, dQ):
    result = np.zeros(nF)
    nSteps = np.shape(tau_sa)[1]
    for z in range(1,nSteps):
        prod_pi = 1
        for k in range(1,nSteps):
            if k != z:
                s_k = tau_sa[k,0]
                a_k = tau_sa[k,1]
                prod_pi *= pi_sto[s_k,a_k]
        s_z = tau_sa[z,0]
        a_z = tau_sa[z,1]
        second_term = 0
        for a in range(nA):
            second_term += pi_sto[s_z,a]*dQ[:,s_z*a]
        dpi = pi_sto[s_z,a_z]*(dQ[:,s_z*a_z] - second_term)
        result += dpi*prod_pi
    return result

def calcLogLLH(w, trajInfo, mdp, options):

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

    # Soft-max policy
    pi_sto = np.exp(NBQ)  # Just pi, not log pi anymore

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        llh += n*NBQ[s, a]

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
        j = (a) * nS+s
        grad += n*dlogPi[:, j] 
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