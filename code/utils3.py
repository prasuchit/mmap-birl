import numpy as np
from operator import mod
from scipy import stats
from gridworld import neighbouring
import generator

def sid2vals(s, nOnionLoc = 4, nEEFLoc = 4, nPredict = 3, start = None):
    ''' 
    FOR SORTING PROBLEM
    Given state id, this func converts it to the 3 variable values 
    '''
    sid = s
    onionloc = int(mod(sid, nOnionLoc))
    sid = (sid - onionloc)/nOnionLoc
    eefloc = int(mod(sid, nEEFLoc))
    sid = (sid - eefloc)/nEEFLoc
    predic = int(mod(sid, nPredict))
    if np.sum(start) != None:
        if isValidState(onionloc, eefloc, predic):
                start[s] = 1
        return onionloc, eefloc, predic, start
    else: return onionloc, eefloc, predic


def vals2sid(ol, eefl, pred, nOnionLoc = 4, nEEFLoc = 4, nPredict = 3):
    ''' 
    FOR SORTING PROBLEM
    Given the 3 variable values making up a state, this converts it into state id 
    '''
    return (ol + nOnionLoc * (eefl + nEEFLoc * pred))


def getValidActions(onionLoc, eefLoc, pred):
    ''' 
    Onionloc: {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    Actions: {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion'}
    '''
    if onionLoc == 0:
        if eefLoc == onionLoc:  
            actidx = [4]
        else:
            actidx = [3]
    elif onionLoc == 1:
        if pred == 0:
            actidx = [2]
        elif pred == 1:
            actidx = [1]
        else:
            actidx = [0]
    elif onionLoc == 2:
        actidx = [4]
    elif onionLoc == 3:
        if pred == 2:
            actidx = [0]
        elif pred == 0:
            actidx = [2]
        else:
            actidx = [1]
    return actidx


def findNxtStates(onionLoc, eefLoc, pred, a):
    ''' 
    Onionloc: {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    Actions: {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion'}
    '''
    if a == 0:
        ''' InspectAfterPicking '''
        if pred == 2:  # it can predict claimed-gripped onion only if prediction is unknown
            return [[1, 1, 0], [1, 1, 1]]   # Equally probable states. Could make it more stochastic.
        else:
            return [[1, 1, pred]]
    elif a == 1:
        ''' PlaceOnConveyor '''
        ''' After we attempt to place on conveyor, pred should become unknown '''
        return [[0, 0, 2]]
    elif a == 2:
        ''' PlaceInBin '''
        return [[2, 2, 2]]
    elif a == 3:
        ''' Pick '''
        return [[3, 3, pred]]
    elif a == 4:
        ''' ClaimNewOnion '''
        return [[0, 3, 2]]

def isValidState(onionLoc, eefLoc, pred):
    if (onionLoc == 1 or eefLoc == 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3):
        return False
    return True

def isValidNxtState(a, onionLoc, eefLoc, pred):
    if (onionLoc == 1 and eefLoc != 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3):
        return False
    if a == 1 or a == 2:
        if (onionLoc == 4 and pred != 2) or (onionLoc == 2 and pred != 2):
            return False
    return True

def getKeyFromValue(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def applyObsvProb(problem, policy, mdp):
    ''' 
    @brief  Here we synthetically generate noisy observations
    using simulated true trajectories.
    NOTE: This function was written a LONG time ago and I'm pretty sure
    I can do a much better job if I rewrite it now. Adding to my todo list.
    '''
    if problem.name == 'sorting':
        trajs, _, _ = generator.sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        obsvs = np.copy(trajs)
        i = 0
        for m in range(problem.nTrajs):
            for h in range(problem.nSteps):
                if problem.trajType == 'synthetic':
                    s = int(trajs[m,h,0])
                    onionLoc, eefLoc, pred = sid2vals(s, problem.nOnionLoc, problem.nEEFLoc, problem.nPredict)
                    if pred != 2:
                        # Assumptions: These need to be replaced with real world values later.
                        # prediction that onion is bad. 95% accuracy of detection
                        # 30% of claimable onions on conveyor are bad
                        pp = 0.3*0.95
                        pred = np.random.choice([pred, int(not pred)], 1, p=[1-pp, pp])[0]
                    obsvs[m,h,0] = vals2sid(onionLoc, eefLoc, pred, problem.nOnionLoc, problem.nEEFLoc, problem.nPredict)
                else:
                    trajsSANet = np.loadtxt(os.getcwd()+"\csv_files\trajsFromSANet.csv", dtype = int)
                    obsvs[m,h,0] = trajsSANet[i]
                    obsvs[m,h,1] = policy[trajsSANet[i]]
                    i += 1
        # print("Hello")
    elif problem.name == 'gridworld':
        nS = mdp.nStates
        trajs, _, _ = generator.sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        obsvs = np.copy(trajs)
        for m in range(problem.nTrajs):
            for h in range(problem.nSteps):
                s = int(trajs[m,h,0])
                if s == 15:
                    pp = 0.3    # With a 30% chance we see the agent in 14th state if he's in the goal state.
                    s = np.random.choice([s, 14], 1, p=[1-pp, pp])[0]
                obsvs[m,h,0] = s

    return obsvs

def getObsvInfo(obsvs, mdp):
    '''
    @brief Generates the observation model. 
    Could be made more elaborate in the future.
    '''
    nS = mdp.nStates
    nA = mdp.nActions
    nTraj = np.shape(obsvs)[0]
    nSteps = np.shape(obsvs)[1]
    obsvsCopy = np.copy(obsvs)
    obs_prob = np.zeros((nTraj,nSteps,nS, nA))
    for m in range(nTraj):
        for h in range(nSteps):
            s = obsvsCopy[m,h,0]
            a = obsvsCopy[m,h,1]
            if s != -1:
                if mdp.name == 'sorting':
                    onionLoc, eefLoc, pred = sid2vals(s)
                    s_noisy = vals2sid(onionLoc, eefLoc, int(not pred))
                    if pred != 2:
                        pp = 0.3*0.95
                        obs_prob[m,h,s_noisy,a] = pp
                        obs_prob[m,h,s,a] = 1 - pp
                    else:
                        obs_prob[m,h,s,a] = 1
                elif mdp.name == 'gridworld':
                        if s == 15:
                            pp = 0.3
                            obs_prob[m,h,14,a] = pp # 30% chance that we got state 14 instead.
                            obs_prob[m,h,s,a] = 1 - pp    
                        else:
                            obs_prob[m,h,s,a] = 1
    return obs_prob