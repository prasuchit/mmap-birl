import parameters as params
import generator
import options
# import birl
import utils
import llh
import numpy as np
import copy
import solver
np.seterr(divide='ignore', invalid='ignore')


def main():
    alg = options.algorithm('MAP_BIRL', 'BIRL', 'Gaussian') # Calling the class algorithm inside options and sending args

    irlOpts = params.paramsSEIRL(alg, restart=0, disp=True) # This method is from parameters.py

    #### irlOpts Output ####
    # alg:'MAP_BIRL'
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
    nTrajs = 1
    nSteps = 10
    problemSeed = 1
    problem = params.problemParamsSE(name, nTrajs=nTrajs, nSteps=nSteps, seed=problemSeed)  # Returns the updated values for
                                                                                            # problem.
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
    ###############################

    numOccs = 1
    mdp = generator.generateProblem(problem)    # Returns an MDP with all the parameters set.
    data = generator.generateDemonstration(mdp, problem, numOccs)

    opts = irlOpts

    trajs = data.trajSet

    cache = []

    w0 = utils.sampleNewWeight(mdp.nFeatures, opts)

    print("Compute initial posterior and gradient ...")
    initPost, initGrad = llh.calNegMarginLogPost(w0, trajs, mdp, opts)
    print("Compute initial opimality region ...")
    pi, H = computerOptmRegn(mdp, w0)   # Page 6 Algo 1 steps 2,3 Map inference paper
    print("Cache the results ...")
    cache.append([pi, H, initGrad])

    constraint = np.matmul(H, w0)
    compare = np.where(constraint < 0)
    
    MaxIter = 100
    currWeight = w0
    currGrad = initGrad
    sigma = 0.01
    print("======== MAP Inference ========")
    for i in range(MaxIter):
        print("- %d iter" % (i))
        currWeight += sigma * currGrad
        opti = reuseCacheGrad(currWeight, cache)
        if opti is None:
            print("  No existing cached gradient reusable ")
            pi, H = computerOptmRegn(mdp, currWeight)
            post, currGrad = llh.calNegMarginLogPost(currWeight, trajs, mdp, opts)
            print("Posterior is: ", post)
            cache.append([pi, H, currGrad])
        else:
            print("  Found reusable gradient ")
            currGrad = opti[2]

    finalPost, finalGrad = llh.calNegMarginLogPost(currWeight, trajs, mdp, opts)
    print(initPost)
    print(finalPost)


def reuseCacheGrad(w, cache):
    for opti in cache:
        H = opti[1]
        constraint = np.matmul(H, w)
        compare = np.where(constraint < 0)
        if compare[0].size > 0:
            return opti

    return None


def computerOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    piL, V, Q, H = solver.policyIteration(mdp)
    return piL, H


if __name__ == "__main__":
    main()