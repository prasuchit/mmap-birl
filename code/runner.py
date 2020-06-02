# import birl
import parameters as params
import generator
import options
import utils
import llh
import numpy as np
import copy
import solver
import time
np.seterr(divide='ignore', invalid='ignore')


def main():
    algo = options.algorithm('MAP_BIRL', 'BIRL', 'Gaussian') # Calling the class algorithm inside options and sending args

    irlOpts = params.setIRLParams(algo, restart=0, disp=True) # This method is from parameters.py

    #### irlOpts Output ####
    # algo:'MAP_BIRL'
    # eta:1.0
    # lb:-1
    # llhType:'BIRL'
    # mu:0.0
    # optimizer:'L-BFGS-B'
    # priorType:'Gaussian'
    # restart:0
    # showMsg:True
    # sigma:0.1
    # ub:1
    ########################

    name = 'gridworld'
    nTrajs = 200
    nSteps = 500
    problemSeed = 1
    init_gridSize = 4
    init_blockSize = 1
    init_noise = 0.3
    numOcclusions = 1
    MaxIter = 100
    sigma = 0.01

    problem = params.setProblemParams(name, nTrajs=nTrajs, nSteps=nSteps, gridSize=init_gridSize, blockSize=init_blockSize, noise=init_noise, seed=problemSeed)  
    
    #### Problem values returned #####
    # blockSize:2
    # discount:0.9
    # filename:'gridworld_12x2'
    # gridSize:12
    # initSeed:1
    # iters:array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # nExperts:1
    # nExps:10
    # nSteps:10
    # nTrajs:1
    # name:'gridworld'
    # noise:0.3
    # seed:1
    #################################
    
    mdp = generator.generateMDP(problem)    # Returns an MDP with all the parameters set.

    data = generator.generateDemonstration(mdp, problem, numOcclusions)

    opts = irlOpts

    trajs = data.trajSet

    expertPolicy = data.policy

    print("Sampling a new weight...")
    w0 = utils.sampleNewWeight(mdp.nFeatures, opts)
    
    cache = []

    t0 = time.time()
    print("Compute initial posterior and gradient ...")
    initPost, initGrad = llh.calcNegMarginalLogPost(w0, trajs, mdp, opts)
    print("Compute initial opimality region ...")
    pi, H = computeOptmRegn(mdp, w0)   # Page 6 Algo 1 steps 2,3 Map inference paper
    print("Cache the results ...")
    cache.append([pi, H, initGrad])

    constraint = np.matmul(H, w0)
    compare = np.where(constraint < 0)
    
    ###################################################################
    # This is just copying the reference

    # copy_list = org_list

    # you should use

    # copy_list = org_list[:]    # make a slice that is the whole list

    # or

    # copy_list = list(org_list) 

    # In case of list of lists, use deep copy

    # copy_list = copy.deepcopy(org_list)
    ###################################################################

    currWeight = np.copy(w0)
    currGrad = np.copy(initGrad)

    initWeightDiff = np.linalg.norm(data.weight - currWeight)
    
    print("======== MAP Inference ========")
    for i in range(MaxIter):    # Finding this: δ_t * ∇_R P(R|X)
        print("- %d iter" % (i))
        currWeight += sigma * currGrad
        opti = reuseCacheGrad(currWeight, cache)
        if opti is None:
            print("  No existing cached gradient reusable ")
            pi, H = computeOptmRegn(mdp, currWeight)
            post, currGrad = llh.calcNegMarginalLogPost(currWeight, trajs, mdp, opts)
            print("Posterior is: ", post)
            cache.append([pi, H, currGrad])
        else:
            print("  Found reusable gradient ")
            currGrad = opti[2]

    mapWeightDiff = np.linalg.norm(data.weight - currWeight)
    mdp = utils.convertW2R(currWeight, mdp) # Updating MDP weights after performing MAP inference
    MAPpi, MAPvalue, _, _ = solver.policyIteration(mdp)
    finalPost, finalGrad = llh.calcNegMarginalLogPost(currWeight, trajs, mdp, opts)
    mdp = utils.convertW2R(finalGrad, mdp) # Updating MDP weights after performing MAP inference
    learnedPolicy, learnedValue, _, _ = solver.policyIteration(mdp)
    print("Same number of actions between expert and MAPpi: ",(MAPpi.squeeze()==expertPolicy.squeeze()).sum(),"/",init_gridSize*init_gridSize)
    print("Same number of actions between expert and learned pi: ",(learnedPolicy.squeeze()==expertPolicy.squeeze()).sum(),"/",init_gridSize*init_gridSize)
    print("Expert's Policy: \n",piInterpretation(expertPolicy.squeeze()))
    print("After MAP policy: \n",piInterpretation(MAPpi.squeeze()))
    # print("MAP Value: ", np.mean(MAPvalue))
    print("Learned Policy: \n",piInterpretation(learnedPolicy.squeeze()))
    # print("Learned Value: ", np.mean(learnedValue))
    t1 = time.time()
    finWeightDiff = np.linalg.norm(data.weight - mdp.weight)
    runtime = t1 - t0
    truePost = data.trueReward
    # print("True Reward: ",truePost)
    # print("Learned Reward: ",initPost)
    # print("MAP Reward: ",finalPost)
    # print("MAP weights: \n",finalGrad)
    # print("Total Runtime: ", runtime)

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

def reuseCacheGrad(w, cache):
    for opti in cache:
        H = opti[1]
        constraint = np.matmul(H, w)
        compare = np.where(constraint < 0)
        if compare[0].size > 0:
            return opti

    return None


def computeOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    piL, _, _, H = solver.policyIteration(mdp)
    return piL, H


if __name__ == "__main__":
    main()
