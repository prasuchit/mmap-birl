import numpy as np
import utils
import utils2
import llh
import time
from tqdm import tqdm
from scipy.optimize._minimize import minimize
import solver

class trajNode:
    def __init__(self, s, a, parent):
        self.s = s
        self.a = a
        self.pair = str(s) + ' ' + str(a)
        self.parent = parent

    def __str__(self):
        s = ''
        s += 'sa: ' + str(self.s) + ', ' + str(self.a) + '\n'
        if self.parent is None:
            s += '  root not has no parent'
        else:
            s += '  parent: ' + str(self.parent.s) + ', ' + str(self.parent.a) + '\n'
        return s

def MMAP(data, mdp, opts, logging=True):
    trajs = data.trajSet
    if opts.optimizer is None:
        print('ERR: no opimizer defined.')
        return

    w0 = utils.sampleNewWeight(mdp.nFeatures, opts, data.seed)
    t0 = time.time()
    res = minimize(llh.calcNegMarginalLogPost, w0, args=(trajs, mdp, opts), tol=1e-8, method=opts.optimizer, jac=True, options={'maxiter': 200, 'disp': opts.showMsg})
    t1 = time.time()
    runtime = t1 - t0
    wL = res.x
    logPost = res.fun
    wL = utils2.normalizedW(wL, opts.normMethod)
    rewardDiff, valueDiff, policyDiff, piL, piE = utils2.computeResults(data, mdp, wL)
    print("Time taken: ", runtime," seconds")
    print("Same number of actions between expert and learned pi: ",(piL.squeeze()==piE.squeeze()).sum(),"/",mdp.nStates)
    print("Reward Diff: {}| Value Diff: {}| Policy Diff: {}".format(rewardDiff,valueDiff.squeeze(),policyDiff))
    print("Sampled weights: \n", w0, "\nLearned weights: \n", wL)
    return wL, logPost, runtime


def MAP(data, mdp, opts, logging=True):

    trajs = data.trajSet
    if opts.optimizer is None:
        print('ERR: no opimizer defined.')
        return

    trajInfo = utils.getTrajInfo(trajs, mdp)
    if opts.restart > 0:

        sumtime = 0
        sumLogPost = 0
        for i in tqdm(range(opts.restart)):
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts, data.seed)
            t0 = time.time()
            res = minimize(llh.calcNegLogPost, w0, args=(trajInfo, mdp, opts), tol=1e-8, method=opts.optimizer, jac=True, options={'maxiter': 5, 'disp': opts.showMsg})
            t1 = time.time()
            sumtime += t1 - t0
            wL = res.x
            logPost = res.fun
            sumLogPost += logPost
        runtime = sumtime / opts.restart
        logPost = sumLogPost / opts.restart
        print(res)
    else:
        t0 = time.time()
        res = minimize(llh.calcNegLogPost, w0, args=(trajInfo, mdp, opts), method=opts.optimizer, jac=True, options={'disp': opts.showMsg})
        t1 = time.time()
        
        runtime = t1 - t0
        wL = res.x
        logPost = res.fun
        
    wL = utils2.normalizedW(wL, opts.normMethod)
    rewardDiff, valueDiff, policyDiff, piL, piE = utils2.computeResults(data, mdp, wL)
    print("Time taken: ", runtime," seconds")
    print("Same number of actions between expert and learned pi: ",(piL.squeeze()==piE.squeeze()).sum(),"/",mdp.nStates)
    print("Reward Diff: {}| Value Diff: {}| Policy Diff: {}".format(rewardDiff,valueDiff.squeeze(),policyDiff))
    print("Sampled weights: \n", w0, "\nLearned weights: \n", wL)
    return wL, logPost, runtime