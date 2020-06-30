import numpy as np
import utils
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
    # w0 = data.weight
    # initPost, _ = llh.calcNegMarginalLogPost(w0, trajs, mdp, opts)
    t0 = time.time()
    res = minimize(llh.calcNegMarginalLogPost, w0, args=(trajs, mdp, opts), method=opts.optimizer, jac=True, options={'disp': opts.showMsg})
    t1 = time.time()
    runtime = t1 - t0
    wL = res.x
    logPost = res.fun

    mdp = utils.convertW2R(data.weight, mdp)
    piE, VE, QE, HE = solver.policyIteration(mdp)
    vE = np.matmul(np.matmul(data.weight.T,HE.T),mdp.start)

    mdp = utils.convertW2R(wL, mdp)
    piL, VL, QL, HL = solver.policyIteration(mdp)
    vL = np.matmul(np.matmul(wL.T,HL.T),mdp.start)

    d  = np.zeros((mdp.nStates, 1))
    for s in range(mdp.nStates):
        ixE = QE[s, :] == max(QE[s, :])
        ixL = QL[s, :] == max(QL[s, :])
        if ((ixE == ixL).all()):
            d[s] = 0
        else:
            d[s] = 1

    wL = (wL-min(wL))/(max(wL)-min(wL))
    rewardDiff = np.linalg.norm(data.weight - wL)
    valueDiff  = abs(vE - vL)
    policyDiff = np.sum(d)/mdp.nStates
    print("Reward Diff: {}| Value Diff: {}| Policy Diff: {}".format(rewardDiff,valueDiff.squeeze(),policyDiff))
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
            # initPost, _ = llh.calcNegLogPost(w0, trajInfo, mdp, opts)
            t0 = time.time()
            res = minimize(llh.calcNegLogPost, w0, args=(trajInfo, mdp, opts), tol=1e-8, method=opts.optimizer, jac=True, options={'disp': opts.showMsg})
            t1 = time.time()
            sumtime += t1 - t0
            wL = res.x
            logPost = res.fun
            sumLogPost += logPost
        runtime = sumtime / opts.restart
        logPost = sumLogPost / opts.restart
        # print(w0)
    else:
        t0 = time.time()
        res = minimize(llh.calcNegLogPost, w0, args=(trajInfo, mdp, opts), method=opts.optimizer, jac=True, options={'disp': opts.showMsg})
        t1 = time.time()
        
        runtime = t1 - t0
        wL = res.x
        logPost = res.fun
        # print(w0)
        mdp = utils.convertW2R(data.weight, mdp)

    piE, VE, QE, HE = solver.policyIteration(mdp)
    vE = np.matmul(np.matmul(data.weight.T,HE.T),mdp.start)

    mdp = utils.convertW2R(wL, mdp)
    piL, VL, QL, HL = solver.policyIteration(mdp)
    vL = np.matmul(np.matmul(wL.T,HL.T),mdp.start)

    d  = np.zeros((mdp.nStates, 1))
    for s in range(mdp.nStates):
        ixE = QE[s, :] == max(QE[s, :])
        ixL = QL[s, :] == max(QL[s, :])
        if ((ixE == ixL).all()):
            d[s] = 0
        else:
            d[s] = 1

    wL = (wL-min(wL))/(max(wL)-min(wL))
    rewardDiff = np.linalg.norm(data.weight - wL)
    valueDiff  = abs(vE - vL)
    policyDiff = np.sum(d)/mdp.nStates
    print("Reward Diff: {}| Value Diff: {}| Policy Diff: {}".format(rewardDiff,valueDiff.squeeze(),policyDiff))
    return wL, logPost, runtime