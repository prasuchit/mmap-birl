import birl
import parameters as params
import generator
import options
import utils
import utils2
import utils3
import llh
import numpy as np
import copy
import solver
import time
import os
import yaml
import logging
np.seterr(divide='ignore', invalid='ignore')
'''
@author Prasanth Suresh
@email  ps32611@uga.edu
@brief  This algorithm extends MAP BIRL(2011) by J.Choi et al. to work
        with input data that contains noise and missing pieces.
        The results are tested on a formative toy problem and a summative
        robotic sorting domain. This paper has been published in UAI 2022 conference.
        PDF currently available at https://arxiv.org/pdf/2109.07788.pdf. 
'''
logger = logging.getLogger(__name__)
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

def main():

    # Read YAML file
    init_yaml_file = open(PACKAGE_PATH+"\yaml_files\init_params.yaml")       

    data_loaded = yaml.safe_load(init_yaml_file)

    irlOpts = params.setIRLParams(data_loaded)

    problem = params.setProblemParams(data_loaded)

    mdp = generator.generateMDP(problem)

    expertData = generator.generateDemonstration(mdp, problem)

    opts = irlOpts

    trajs = expertData.trajSet

    # expertPolicy = expertData.policy

    if(opts.solverMethod == 'scipy'):
        logger.warning(
            f"NOTE: scipy solver hasn't been updated and tested with all the latest changes of MMAP.\
                Results may not be accurate. I strongly suggest using the manual method!"
                )
        if opts.alg == 'MMAP_BIRL':
            print("Calling MMAP BIRL")
            birl.MMAP(expertData, mdp, opts, problem)
        elif opts.alg == 'MAP_BIRL':
            print("Calling MAP BIRL")
            birl.MAP(expertData, mdp, opts)
        else:
            print('Incorrect algorithm chosen: ', opts.alg)

    elif(opts.solverMethod == 'manual'):

        while(opts.restart != 0):

            print("Sampling a new weight...")
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts, problem.seed)

            cache = []

            t0 = time.time()
            print("Compute initial posterior and gradient ...")
            _, initGrad = llh.calcNegMarginalLogPost(
                w0, trajs, mdp, opts, problem)
            print("Compute initial opimality region ...")
            pi, H = utils2.computeOptmRegn(mdp, w0)
            print("Cache the results ...")
            cache.append([pi, H, initGrad])
            currWeight = np.copy(w0)
            currGrad = np.copy(initGrad)

            if opts.optimMethod == 'gradAsc':
                wL = utils2.gradientDescent(
                    mdp, trajs, problem, opts, currWeight, currGrad,  cache)
            elif opts.optimMethod == 'nesterovGrad':
                wL, mdp = utils2.nesterovAccelGrad(
                    mdp, trajs, problem, opts, currWeight, currGrad, cache=cache)

            wL = utils2.normalizedW(wL, opts.normMethod)

            rewardDiff, valueDiff, policyDiff, piL, piE = utils2.computeResults(
                expertData, mdp, wL)

            if(policyDiff >= 0.5 or valueDiff > 0.5):
                print(
                    f"Rerunning for better results!\nValue Diff: {valueDiff.squeeze()} | Policy misprediction: {policyDiff} | Reward Difference: {rewardDiff}")
                opts.restart += 1
                if(opts.restart > 1):
                    print(f"Restarted {opts.restart} times already! Exiting!")
                    exit(0)
            else:
                opts.restart = 0
                print("True weights: \n", expertData.weight,
                      "\nSampled weights: \n", w0, "\nLearned weights: \n", wL)
                t1 = time.time()
                runtime = t1 - t0
                print("Time taken: ", runtime, " seconds")
                print(
                    f"Policy Diff: {policyDiff} | Reward Diff: {rewardDiff}| Value Diff: {valueDiff.squeeze()}")

    else:
        print("Please check your input!")

if __name__ == "__main__":
    main()