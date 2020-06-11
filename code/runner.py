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
from tqdm.gui import tqdm
np.seterr(divide='ignore', invalid='ignore')


def main():

    # choice = input("Enter the method for optimization: scipy or manual\n")
    choice = 'scipy'

    algo = options.algorithm('MAP_BIRL', 'BIRL', 'Gaussian')

    irlOpts = params.setIRLParams(algo, restart=1, optiMethod=choice, disp=True)
    
    name = 'gridworld'
    nTrajs = 200
    nSteps = 500
    problemSeed = 1
    init_gridSize = 10
    init_blockSize = 2
    init_noise = 0.3
    numOcclusions = 1
    MaxIter = 100
    sigma = 0.01

    problem = params.setProblemParams(name, nTrajs=nTrajs, nSteps=nSteps, gridSize=init_gridSize, blockSize=init_blockSize, noise=init_noise, seed=problemSeed)  
    
    mdp = generator.generateMDP(problem)
    
    data = generator.generateDemonstration(mdp, problem, numOcclusions)

    opts = irlOpts

    trajs = data.trajSet

    expertPolicy = data.policy
    
    if(opts.optiMethod == 'scipy'):

        print("Calling MMAP BIRL")
        wL, logPost, runtime = birl.MMAP(data, mdp, opts)
        print("Time taken: ", runtime," seconds")
        mdp = utils.convertW2R(wL, mdp) # Updating learned weights

        learnedPolicy, learnedValue, _, _ = solver.policyIteration(mdp)
        print("Same number of actions between expert and learned pi: ",(learnedPolicy.squeeze()==expertPolicy.squeeze()).sum(),"/",init_gridSize*init_gridSize)
        print("Learned Policy: \n",piInterpretation(learnedPolicy.squeeze()))

    elif(opts.optiMethod == 'manual'):
        

        while(opts.restart == 1):

            print("Sampling a new weight...")
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts)
            
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
                weightUpdate = (sigma * currGrad)
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
            mdp = utils.convertW2R(currWeight, mdp) # Updating learned weights
            learnedPolicy, learnedValue, _, _ = solver.policyIteration(mdp)
            err = mdp.nStates - (learnedPolicy.squeeze()==expertPolicy.squeeze()).sum()
            if(err < mdp.nStates/ 4):
                opts.restart = 0
                #Normalized weights
                wL = (currWeight-min(currWeight))/(max(currWeight)-min(currWeight))
            else:
                #Normalized weights
                # w0 = (currWeight-min(currWeight))/(max(currWeight)-min(currWeight))
                print('Num of values diff from expert: {} / {}'.format(err, mdp.nStates))
                print("Rerunning for better results!")
        print("Same number of actions between expert and learned pi: ",(learnedPolicy.squeeze()==expertPolicy.squeeze()).sum(),"/",init_gridSize*init_gridSize)
        # print("Expert's Policy: \n",piInterpretation(expertPolicy.squeeze()))
        # print("Learned Policy: \n",piInterpretation(learnedPolicy.squeeze()))
        t1 = time.time()
        runtime = t1 - t0
        print("Time taken: ", runtime," seconds")

    else:
        print("Please check your input!")

    mdp = utils.convertW2R(data.weight, mdp)
    piE, VE, QE, HE = solver.policyIteration(mdp)
    vE = np.matmul(np.matmul(data.weight.T,HE.T),mdp.start)

    mdp = utils.convertW2R(wL, mdp)
    piL, VL, QL, HL = solver.policyIteration(mdp)
    vL =np.matmul(np.matmul(wL.T,HL.T),mdp.start)

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
    print("Reward Diff: {}| Value Diff: {}| Policy Diff: {}".format(rewardDiff,valueDiff.squeeze(),policyDiff))
###########################################################################################

def computeOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    piL, _, _, H = solver.policyIteration(mdp)
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
