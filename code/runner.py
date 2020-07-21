import birl
import parameters as params
import generator
import options
import utils
import llh
import numpy as np
import copy
import solver
import time
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp
from tqdm.gui import tqdm
np.seterr(divide='ignore', invalid='ignore')


def main():

    # choice = input("Enter the method for optimization: scipy or manual\n")
    choice = 'manual'
    # choice = 'scipy'
    algoName = 'MAP_BIRL'
    # algoName = 'MMAP_BIRL'
    llhName = 'BIRL'
    priorName = 'Gaussian'
    # priorName = 'Uniform'
    probName = 'highway'
    # probName = 'gridworld'
    nTrajs = 5
    nSteps = 10
    problemSeed = 1
    init_gridSize = 4
    init_blockSize = 1
    init_nLanes = 3     # For highway problem
    init_nSpeeds = 2    # For highway problem
    init_noise = 0.3
    numOcclusions = 1
    MaxIter = 100
    sigma = 1/MaxIter
    alpha = 1   # learning rate
    decay = .95

    algo = options.algorithm(algoName, llhName, priorName)

    irlOpts = params.setIRLParams(algo, restart=1, optiMethod=choice, disp=True)

    problem = params.setProblemParams(probName, nTrajs=nTrajs, nSteps=nSteps, gridSize=init_gridSize, blockSize=init_blockSize, nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)  
    
    mdp = generator.generateMDP(problem)
    
    data = generator.generateDemonstration(mdp, problem, numOcclusions)

    opts = irlOpts

    trajs = data.trajSet

    expertPolicy = data.policy
    
    if(opts.optiMethod == 'scipy'):
        if opts.alg == 'MMAP_BIRL':
            print("Calling MMAP BIRL")
            wL, logPost, runtime = birl.MMAP(data, mdp, opts)
            print("Learned weights: \n", wL)
            print("Time taken: ", runtime," seconds")

        elif opts.alg == 'MAP_BIRL':
            print("Calling MAP BIRL")
            wL, logPost, runtime = birl.MAP(data, mdp, opts)
            print("Learned weights: \n", wL)
            print("Time taken: ", runtime," seconds")
        else:
            print('Incorrect algorithm chosen: ', opts.alg)

    elif(opts.optiMethod == 'manual'):
        

        while(opts.restart != 0):
            print("Sampling a new weight...")
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts, problemSeed)
            
            cache = []

            t0 = time.time()
            
            print("Compute initial posterior and gradient ...")
            initPost, initGrad = llh.calcNegMarginalLogPost(w0, trajs, mdp, opts)
            print("Compute initial opimality region ...")
            pi, H = computeOptmRegn(mdp, w0)
            print("Cache the results ...")
            cache.append([pi, H, initGrad])
            currWeight = np.copy(w0)
            currGrad = np.copy(initGrad)
            
            print("======== MAP Inference ========")
            for i in range(MaxIter):    # Finding this: R_new = R + δ_t * ∇_R P(R|X)
                print("- %d iter" % (i))
                weightUpdate = (sigma * alpha * currGrad)
                alpha *= decay
                weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
                currWeight = currWeight + weightUpdate
                opti = reuseCacheGrad(currWeight, cache)
                if opti is None:
                    print("  No existing cached gradient reusable ")
                    pi, H = computeOptmRegn(mdp, currWeight)
                    post, currGrad = llh.calcNegMarginalLogPost(currWeight, trajs, mdp, opts)
                    # print("Posterior is: ", post)
                    cache.append([pi, H, currGrad])
                else:
                    print("  Found reusable gradient ")
                    currGrad = opti[2]

            wL = (np.exp(currWeight))/(np.sum(np.exp(currWeight))) # Softmax normalization
            # wL = (currWeight-min(currWeight))/(max(currWeight)-min(currWeight)) # Normalizing b/w 0-1
            # wL = currWeight # Unnormalized raw weights
            mdp = utils.convertW2R(data.weight, mdp)
            piE, VE, QE, HE = solver.piMDPToolbox(mdp)
            vE = np.matmul(np.matmul(data.weight.T,HE.T),mdp.start)

            mdp = utils.convertW2R(wL, mdp)
            piL, VL, QL, HL = solver.piMDPToolbox(mdp)
            vL = np.matmul(np.matmul(wL.T,HL.T),mdp.start)

            d  = np.zeros((mdp.nStates, 1))
            for s in range(mdp.nStates):
                ixE = QE[s, :] == max(QE[s, :])
                ixL = QL[s, :] == max(QL[s, :])
                if ((ixE == ixL).all()):
                    d[s] = 0
                else:
                    d[s] = 1

            rewardDiff = np.linalg.norm(data.weight - wL)
            valueDiff  = abs(vE - vL)
            policyDiff = np.sum(d)/mdp.nStates

            # if(policyDiff > 0.15 or rewardDiff > 1.5):
            if(policyDiff > 0.15):
                print(f"Rerunning for better results! Policy misprediction: {policyDiff} | Reward Difference: {rewardDiff}")
                opts.restart += 1
                if(opts.restart > 5):
                    print("Restarted 5 times already! Exiting!")
                    exit(0)
            else:
                opts.restart = 0

        # print("Expert's Policy: \n",piInterpretation(expertPolicy.squeeze()))
        # print("Learned Policy: \n",piInterpretation(learnedPolicy.squeeze()))
        # print("Sampled weights: \n", w0)
        print("Learned weights: \n", wL)
        t1 = time.time()
        runtime = t1 - t0
        print("Same number of actions between expert and learned pi: ",(piL.squeeze()==piE.squeeze()).sum(),"/",mdp.nStates)
        print("Time taken: ", runtime," seconds")
        print(f"Reward Diff: {rewardDiff}| Value Diff: {valueDiff.squeeze()}| Policy Diff: {policyDiff}")

    else:
        print("Please check your input!")

###########################################################################################

def computeOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    piL, _, _, H = solver.piMDPToolbox(mdp)
    return piL, H

def reuseCacheGrad(w, cache):
    for opti in cache:
        H = opti[1]
        constraint = np.matmul(H, w)
        compare = np.where(constraint < 0)
        if compare[0].size > 0:
            return opti

    return None

def piInterpretation(policy):
    actions = {}
    for i in range(len(policy)):
        if(policy[i] == 0):
            actions[i] = 'North'
        elif(policy[i] == 1):
            actions[i] = 'East'
        elif(policy[i] == 2):
            actions[i] = 'West'
        elif(policy[i] == 3):
            actions[i] = 'South'
    return actions


if __name__ == "__main__":
    main()