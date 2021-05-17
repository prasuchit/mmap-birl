import numpy as np
from operator import mod
from scipy import stats
from gridworld import neighbouring
import generator

def sid2vals(s, nOnionLoc = 4, nEEFLoc = 4, nPredict = 3, nlistIDStatus = 3, start = None):
    ''' Given state id, this func converts it to the 4 variable values '''
    sid = s
    onionloc = int(mod(sid, nOnionLoc))
    sid = (sid - onionloc)/nOnionLoc
    eefloc = int(mod(sid, nEEFLoc))
    sid = (sid - eefloc)/nEEFLoc
    predic = int(mod(sid, nPredict))
    sid = (sid - predic)/nPredict
    listidstatus = int(mod(sid, nlistIDStatus))
    if np.sum(start) != None:
        if (listidstatus == 2):
            if isValidState(onionloc, eefloc, predic, listidstatus):
                start[0,s] = 1
        else:
            if isValidState(onionloc, eefloc, predic, listidstatus):
                start[1,s] = 1
        return onionloc, eefloc, predic, listidstatus, start
    else: return onionloc, eefloc, predic, listidstatus


def vals2sid(ol, eefl, pred, listst, nOnionLoc = 4, nEEFLoc = 4, nPredict = 3, nlistIDStatus = 3):
    ''' Given the 4 variable values making up a state, this converts it into state id '''
    return (ol + nOnionLoc * (eefl + nEEFLoc * (pred + nPredict * listst)))


def getValidActions(onionLoc, eefLoc, pred, listidstatus):
    ''' 
    Onionloc: {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    listIDstatus = {0: 'Empty', 1: 'Not Empty', 2: 'Unavailable'} 
    Actions: {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 6: 'ClaimNextInList'}
    '''

    if onionLoc == 0:
        if listidstatus == 2:
            if eefLoc == onionLoc:  
                actidx = [4]
            else:
                actidx = [3]
        elif listidstatus == 0:
                actidx = [5]
        else:
            if pred == 2:
                actidx = [6]
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
        if listidstatus == 2:
            actidx = [4]
        elif listidstatus == 0:
            actidx = [5]
        else:
            actidx = [6]
    elif onionLoc == 3:
        if pred == 2:
            actidx = [0]
        elif pred == 0:
            actidx = [2]
        else:
            actidx = [1]
    return actidx


def findNxtStates(onionLoc, eefLoc, pred, listidstatus, a):
    ''' 
    Onionloc: {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    listIDstatus = {0: 'Empty', 1: 'Not Empty', 2: 'Unavailable'} 
    Actions: {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 6: 'ClaimNextInList'}
    '''

    if a == 0:
        ''' InspectAfterPicking '''
        # Assumptions: These need to be replaced with real world values later.
        # prediction that onion is bad. 95% accuracy of detection
        # 50% of claimable onions on conveyor are bad
        if pred == 2:  # it can predict claimed-gripped onion only if prediction is unknown
            # pp = 0.5*0.95
            # pred = np.random.choice([1, 0], 1, p=[1-pp, pp])[0]
            return [[1, 1, 0, 2], [1, 1, 1, 2]]
        else:
            return [[1, 1, pred, 2]]
    elif a == 1:
        ''' PlaceOnConveyor '''
        ''' After we attempt to place on conveyor, pred should become unknown '''
        return [[0, 0, 2, 2]]
    elif a == 2:
        ''' PlaceInBin '''
        if listidstatus == 1:
            # pp = 1-(2/numObjects)
            # listidstatus = np.random.choice([1, 0], 1, p=[pp, 1-pp])[0]
            ''' After we attempt to place in bin, pred should become unknown '''
            return [[2, 2, 2, 0], [2, 2, 2, 1]]
        else:
            return [[2, 2, 2, listidstatus]]
    elif a == 3:
        ''' Pick '''
        return [[3, 3, pred, listidstatus]]
    elif a == 4:
        ''' ClaimNewOnion '''
        return [[0, 3, 2, 2]]
    elif a == 5:
        ''' InspectWithoutPicking '''
        # cannot apply this action if a list is already available
        # It is detecting many onions simultaneously. assuming half are bad with 95% probability,
        # it should be derived from chance of not detecting any of bad onions = 0.3^(numObjects/2) .
        # Then prob is 0.95*(1-0.3^(numObjects/2)) ~ 1
        # pp = 0.95*(1 - pow((1-0.7), (numObjects/2)))
        # ls = np.random.choice([1, 0], 1, p=[pp, 1-pp])[0]
        # if (ls == 0):
        #     pred = 2
        # else:
        #     pred = 0
        return [[0, eefLoc, 0, 1], [0, eefLoc, 2, 0]]
    else:
        ''' ClaimNextInList '''
        if listidstatus == 1:
            # if list not empty, then
            return [[0, eefLoc, 0, 1]]
        else:
            # else make onion unknown and list not available
            return [[0, eefLoc, 2, listidstatus]]
    return

def isValidState(onionLoc, eefLoc, pred, listidstatus):
    if listidstatus == 2:
        if (onionLoc == 1 and eefLoc != 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3):
            return False
    else:
        if (onionLoc == 1 or eefLoc == 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3):
            return False
    return True


def isValidNxtState(a, onionLoc, eefLoc, pred, listidstatus):
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

def applyObsvProb(problem,policy,mdp):
    ''' @brief  Here we synthetically generate noisy observations
    using simulated true trajectories.'''
    if problem.name == 'sorting':
        # trajsSANet = np.loadtxt("trajsFromSANet.csv", dtype = int)
        trajs, _, _ = generator.sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed, problem.sorting_behavior)
        obsvs = np.copy(trajs)
        for m in range(problem.nTrajs):
            for h in range(problem.nSteps):
                s = int(trajs[m,h,0])
                onionLoc, eefLoc, pred, listIDStatus = sid2vals(s, problem.nOnionLoc, problem.nEEFLoc, problem.nPredict, problem.nlistIDStatus)
                if pred != 2:
                    # Assumptions: These need to be replaced with real world values later.
                    # prediction that onion is bad. 95% accuracy of detection
                    # 30% of claimable onions on conveyor are bad
                    pp = 0.3*0.95
                    pred = np.random.choice([pred, int(not pred)], 1, p=[1-pp, pp])[0]
                obsvs[m,h,0] = vals2sid(onionLoc, eefLoc, pred, listIDStatus, problem.nOnionLoc, problem.nEEFLoc, problem.nPredict, problem.nlistIDStatus)
    
    elif problem.name == 'gridworld':
        nS = mdp.nStates
        trajs, _, _ = generator.sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed, problem.sorting_behavior)
        obsvs = np.copy(trajs)
        for m in range(problem.nTrajs):
            for h in range(problem.nSteps):
                s = int(trajs[m,h,0])
                if s == 15:
                    pp = 0.2    # With a 40% chance we see the agent in 14th state if he's in the goal state.
                    s = np.random.choice([s, 14], 1, p=[1-pp, pp])[0]
                obsvs[m,h,0] = s

    return obsvs

def getObsvInfo(obsvs, mdp):
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
                    onionLoc, eefLoc, pred, listIDStatus = sid2vals(s)
                    s_noisy = vals2sid(onionLoc, eefLoc, int(not pred), listIDStatus)
                    if pred != 2:
                        pp = 0.3*0.95
                        obs_prob[m,h,s_noisy,a] = pp
                        obs_prob[m,h,s,a] = 1 - pp
                    else:
                        obs_prob[m,h,s,a] = 1
                elif mdp.name == 'gridworld':
                        if s == 15:
                            pp = 0.2
                            obs_prob[m,h,14,a] = pp # 40% chance that we got state 14 instead.
                            obs_prob[m,h,s,a] = 1 - pp    
                        else:
                            obs_prob[m,h,s,a] = 1
    return obs_prob