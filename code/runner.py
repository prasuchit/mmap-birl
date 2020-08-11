import birl
import parameters as params
import generator
import options
import utils
import utils2
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

    solverMethod = 'manual'
    # solverMethod = 'scipy'
    algoName = 'MAP_BIRL'
    # algoName = 'MMAP_BIRL'
    llhName = 'BIRL'
    priorName = 'Gaussian'
    # priorName = 'Uniform'
    # probName = 'highway'
    probName = 'gridworld'
    # optimMethod = 'gradAsc'
    optimMethod = 'nesterovGrad'
    nTrajs = 10
    nSteps = 50
    problemSeed = 1
    init_gridSize = 4
    init_blockSize = 2
    init_nLanes = 3     # Highway problem
    init_nSpeeds = 2    # Highway problem
    init_noise = 0.3
    numOcclusions = 1
    useSparse = 0

    normMethod = 'None'  # 'softmax' '0-1' 'None'

    algo = options.algorithm(algoName, llhName, priorName)

    irlOpts = params.setIRLParams(algo, restart=1, solverMethod=solverMethod, optimMethod = optimMethod, normMethod = normMethod, disp=True)

    problem = params.setProblemParams(probName, nTrajs=nTrajs, nSteps=nSteps, nOccs = numOcclusions,  gridSize=init_gridSize, 
            blockSize=init_blockSize, nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed, useSparse = useSparse)  
    
    mdp = generator.generateMDP(problem)
    
    expertData = generator.generateDemonstration(mdp, problem, problem.nOccs)

    opts = irlOpts

    trajs = expertData.trajSet

    expertPolicy = expertData.policy
    
    if(opts.solverMethod == 'scipy'):

        if opts.alg == 'MMAP_BIRL':
            print("Calling MMAP BIRL")
            birl.MMAP(expertData, mdp, opts)
        elif opts.alg == 'MAP_BIRL':
            print("Calling MAP BIRL")
            birl.MAP(expertData, mdp, opts)
        else:
            print('Incorrect algorithm chosen: ', opts.alg)

    elif(opts.solverMethod == 'manual'):
        
        while(opts.restart != 0):
            
            w0 = None

            print("Sampling a new weight...")
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts, problemSeed)
            
            cache = []

            t0 = time.time()
            
            print("Compute initial posterior and gradient ...")
            initPost, initGrad = llh.calcNegMarginalLogPost(w0, trajs, mdp, opts)
            print("Compute initial opimality region ...")
            pi, H = utils2.computeOptmRegn(mdp, w0)
            print("Cache the results ...")
            cache.append([pi, H, initGrad])
            currWeight = np.copy(w0)
            currGrad = np.copy(initGrad)
            
            if optimMethod == 'gradAsc':
                wL = utils2.gradientDescent(mdp, trajs, opts, currWeight, currGrad, cache)
            elif optimMethod == 'nesterovGrad':
                wL = utils2.nesterovAccelGrad(mdp, trajs, opts, currWeight, currGrad, cache = cache)
            
            wL = utils2.normalizedW(wL, normMethod)

            rewardDiff, valueDiff, policyDiff, piL, piE = utils2.computeResults(expertData, mdp, wL)

            if(policyDiff > 0.2 or valueDiff > 3):
            # if(valueDiff > 5):
                print(f"Rerunning for better results!\nValue Diff: {valueDiff.squeeze()} | Policy misprediction: {policyDiff} | Reward Difference: {rewardDiff}")
                opts.restart += 1
                if(opts.restart > 15):
                    print(f"Restarted {opts.restart} times already! Exiting!")
                    exit(0)
            else:
                # print("Expert's Policy: \n",utils2.piInterpretation(expertPolicy.squeeze(), problem.name))
                # print("Learned Policy: \n",utils2.piInterpretation(learnedPolicy.squeeze(), problem.name))
                # print("Sampled weights: \n", w0)
                opts.restart = 0
                print("Sampled weights: \n", w0, "\nLearned weights: \n", wL)
                t1 = time.time()
                runtime = t1 - t0
                print("Same number of actions between expert and learned pi: ",(piL.squeeze()==piE.squeeze()).sum(),"/",mdp.nStates)
                print("Time taken: ", runtime," seconds")
                print(f"Policy Diff: {policyDiff} | Reward Diff: {rewardDiff}| Value Diff: {valueDiff.squeeze()}")

    else:
        print("Please check your input!")

if __name__ == "__main__":
    main()