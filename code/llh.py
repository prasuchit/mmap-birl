import numpy as np
import utils
import solver
import math
import copy
from tqdm import tqdm
from scipy.special._logsumexp import logsumexp
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')


def calcNegMarginalLogPost(w, trajs, mdp, options):
# Calculate Negative Marginalized Log Posterior
# Check page 6 line 5 choi-kim paper for concept
    originalInfo = utils.getTrajInfo(trajs, mdp)
    occs = originalInfo.occlusions

    llh = 0
    grad1 = 0
    trajsCopy = copy.copy(trajs)
    if(-1 in trajsCopy):
        print("Compute posterior with marginalization...")
        for s in tqdm(range(mdp.nStates)):
            for a in range(mdp.nActions):
                # occs is a 2D array containing [m,h] pairs
                # where  m = trajectory count; h = step count;
                # So for instance occs[0,0] = m and occs[0,1] = h
                trajsCopy[occs[0,0], occs[0,1], 0] = s
                trajsCopy[occs[0,0], occs[0,1], 1] = a

                # What we are doing above is for the trajectory index and step index where we found the occlusion
                # we are creating nS,nA number of new trajectories to marginalize them by considering that occluded 
                # location to be each state and that every action is performed in that state. For every such new 
                # trajectory copy created, we get its info from utils function and find the marginal log likelihood 
                # and marginal gradient. We find the summation of all of them to marginalize andd finally obtain the 
                # posterior and the corresponding gradient for it.

                trajInfo = utils.getTrajInfo(trajsCopy, mdp)    # For each occluded obsv in a trajectory get info
                # print(f"State {s}, action {a}")
                mllh, mgrad1 = calcLogLLH(w, trajInfo, mdp, options)    # Getting back the log likelihood and gradient
                                                                        # for the mth trajectory.
                # print(f"mllh: {mllh}, mgrad1: {mgrad1}")
                llh += mllh # Adding all the individual trajectory likelihood values
                grad1 += mgrad1
    else:
        print("No occlusions found...")
        trajInfo = utils.getTrajInfo(trajsCopy, mdp)    # For each occluded obsv in a trajectory get info
        # print(f"State {s}, action {a}")
        mllh, mgrad1 = calcLogLLH(w, trajInfo, mdp, options)    # Getting back the log likelihood and gradient
                                                                # for the mth trajectory.
        # print(f"mllh: {mllh}, mgrad1: {mgrad1}")
        llh += mllh # Adding all the individual trajectory likelihood values
        grad1 += mgrad1
    # grad1 is the second term (marginalized) of eq 25 shibo's writeup
    grad1 = (grad1).reshape(mdp.nFeatures,1)
    prior, grad2 = calcLogPrior(w, options)  
    post = prior + llh
    grad = grad1 + grad2
    # grad = -grad
    # post = -post

    # print(f"posterior:\n {post},\nlog likelihood:\n {llh},\nprior:\n {prior},\ngradient:\n {grad},\ngrad1:\n {grad1},\ngrad2:\n {grad2}")
    return post, grad

def calcNegLogPost(w, trajInfo, mdp, options):
# Calculate Negative Log Posterior    
    llh, grad1 = calcLogLLH(w, trajInfo, mdp, options)
    prior, grad2 = calcLogPrior(w, options)
    post = prior + llh
    grad = grad1 + grad2

    return post, grad

def calcLogPrior(w, options):
# Calculate Log Prior
# Useful link: 
#   https://stats.stackexchange.com/questions/90134/gradient-of-multivariate-gaussian-log-likelihood

    if options.priorType == 'Gaussian':
        # The next few lines just calculate the gaussian distb equation; check Wikipedia
        x = w - options.mu;
        prior = np.sum(np.matmul(np.transpose(x), x) * -1 / 2 * math.pow(options.sigma, 2))
        grad = -x / math.pow(options.sigma, 2)
    else:
        prior = math.log(1)
        grad = np.zeros(w.shape)
        
    return prior, grad

def calcLogLLH(w, trajInfo, mdp, options):
# Calculate Log Likelihood
    mdp = utils.convertW2R(w, mdp)
    piL, VL, QL, H = solver.policyIteration(mdp)    # QL is a 144*4 matrix with Q values in every state for all 4 actions
    dQ = calcGradQ(piL, mdp)    # nFx(nSxnA) matrix
    nF = mdp.nFeatures
    nS = mdp.nStates
    nA = mdp.nActions
    eta = options.eta
#########################################################################################
    BQ = eta * QL   # eta is the inverse temp component of a boltzmann distrib
                    # Which defines the amount of contribution each Q value has in
                    # the agent's learning process. Here we take eta as 1.

    # BQSum = np.log(np.sum(np.exp(BQ), axis=1))  # Also explained in Choi-Kim MAP paper sec 2.3
    
    BQSum = logsumexp(BQ, axis=1)
    # The above two lines are based on BIRL paper by Ramachandran-Amir, section 3.1
    # The gist of that is that by using the e^(Boltzmann Q value) we can find the likelihood
    # of some action being done in some state. log(sum(e^BQ)) gives the log value of sum of
    # all the action likelihood values or the log-likelihood of all the actions in a state

    NBQ = BQ.copy()
    for i in range(nA):
        NBQ[:, i] = NBQ[:, i] - BQSum[:]    # log(map paper Choi-Kim, section 2.3 Eq 4)

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        llh += NBQ[s, a]*n

    # ################### Original code ##########################################
    # pi_sto = np.exp(BQ)
    # pi_sto_sum = np.sum(pi_sto, axis=1)
    # for i in range(nA):
    #     pi_sto[:, i] = pi_sto[:, i] / pi_sto_sum[:] # Eq 26 of shibo's writeup
    # #######################################################################################

    # pi_sto is a nSxnA matrix
    # Soft-max policy
    pi_sto = np.exp(NBQ)    # Eq 26 of shibo's writeup

    # calculate dlogPi/dw
    dlogPi = np.zeros((nF, nS * nA))
    for f in range(nF):
        z = np.reshape(dQ[f, :], (nS, nA))
        # temp = np.sum(pi_sto * np.reshape(dQ[f, :], (nS, nA)), axis=1)[:]
        for i in range(nA):
            z[:, i] = z[:, i] - np.sum(pi_sto * np.reshape(dQ[f, :], (nS, nA)), axis=1)[:]  # Eq 27, last term of shibo's writeup
        # eq27 = pi_sto * z
        dlogPi[f, :] = eta * np.reshape(z, (1, nS * nA))  # Original code
        # dlogPi[f, :] = np.reshape(eq27, (1, nS * nA))  # Eq 27 of shibo's writeup

    # Calculating the gradient of the reward function
    grad = np.zeros(nF)
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        j = a * nS+s;
        grad += n*dlogPi[:, j]    # 1st term inside summation of eq 25 Shibo's writeup
    # print("Grad in log llh: ", grad)    
    return llh, grad
#################################################################################################

def calcGradQ(piL, mdp):
# Calculate Gradient Q value
    nS = mdp.nStates
    nA = mdp.nActions
    Epi = np.zeros((nS, nS * nA)).astype(int)

    """
    To list all possibilities of a state space, from every state, you could end up in every 
    other state by doing any one of the available actions. Epi stores the expectation value
    of ending up in some state by doing some action, as 1.
    Eg: 12x12 grid, 4 actions => 144 states and 144x4 possible next actions from each state. 
    So your matrix would be of the size [nS,nSxnA] => [144,576]
    (forget about reachability constraints, this only lists the possibilities).

    The next line gives the action number you are performing from the 
    current state if you followed the policy action.
    """

    idx = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    
    for i in range(nS):
        # Setting the expectation value for that current state and action as 1
        Epi[i, idx[i]] = 1 

    """dQ equation is provided at the end of the supplementary material of Choi and Kim's 
    MAP for BIRL paper, under theorem 3."""

    dQ = np.linalg.lstsq(np.eye(nS * nA) - np.matmul(mdp.T, Epi), mdp.F)[0]
    return np.transpose(dQ)