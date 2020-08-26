import numpy as np
from operator import mod


def sid2vals(s, nOnionLoc, nEEFLoc, nPredict, nlistIDStatus, start):
    sid = s
    onionloc = int(mod(sid, nOnionLoc))
    sid = (sid - onionloc)/nOnionLoc
    eefloc = int(mod(sid, nEEFLoc))
    sid = (sid - eefloc)/nEEFLoc
    predic = int(mod(sid, nPredict))
    sid = (sid - predic)/nPredict
    listidstatus = int(mod(sid, nlistIDStatus))
    if listidstatus == 2 or predic == 1:
        start[s] = 1.0
    return onionloc, eefloc, predic, listidstatus, start


def vals2sid(ol, eefl, pred, listst, nOnionLoc, nEEFLoc, nPredict, nlistIDStatus):
    return (ol + nOnionLoc * (eefl + nEEFLoc * (pred + nPredict * listst)))


def getValidActions(onionLoc, eefLoc, pred, listidstatus):
    ''' 
    Onionloc: {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome', 4: 'Placed'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    listIDstatus = {0: 'Empty', 1: 'Not Empty', 2: 'Unavailable'} 
    Actions: {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 6: 'ClaimNextInList'}'''

    if onionLoc == 0:
        if eefLoc == 0:
            actidx = [3]
        else:
            if listidstatus == 2:  # Cannot claim from list if list not available
                actidx = [4, 5]
            else:
                actidx = [6]
    elif onionLoc == 1 or onionLoc == 3:
        if pred == 0:
            actidx = [2]
        elif pred == 1:
            actidx = [1]
        else:
            actidx = [0]
    elif onionLoc == 2:
        if listidstatus == 2:  # sorter can claim new onion only when a list of predictions has not been pending
            actidx = [4, 5]
        else:
            # We can't allow ClaimNewOnion with a list available
            actidx = [6]
    elif onionLoc == 4:
        if listidstatus == 2:  # Cannot claim from list if list not available
            actidx = [4, 5]
        else:
            actidx = [6]

    # if onionLoc == 1 or onionLoc == 3:
    #     # home or front
    #     actidx = [0,1,2]
    # elif onionLoc == 0 or onionLoc == 4:
    #     # on conveyor (not picked yet or already placed)
    #     if listidstatus == 2:  # cannot claim from list if list not available
    #         actidx = [3,4,5]
    #     else:  # cannot create list again if a list is already available
    #         # if we allow ClaimNewOnion with a list available
    #         actidx = [3,6]
    #         # then it will do *,0,2,1 ClaimNewOnion 0,2,2,1 ClaimNextInList 0,0,2,1
    #         # and will assume onion is bad without inspection
    # elif onionLoc == 2:   # in bin, can't pick from bin because not reachable
    #     if listidstatus == 2:  # sorter can claim new onion only when a list of predictions has not been pending
    #         actidx = [4,5]
    #     else:
    #         # We can't allow ClaimNewOnion with a list available
    #         actidx = [6]

    return actidx


def findNxtStates(onionLoc, eefLoc, pred, listidstatus, a, numObjects):
    # np.random.seed(1)
    if a == 0:
        ''' InspectAfterPicking '''
        # Assumptions: These need to be replaced with real world values later.
        # prediction that onion is bad. 95% accuracy of detection
        # 50% of claimable onions on conveyor are bad
        if pred == 2:  # it can predict claimed-gripped onion only if prediction is unknown
            # pp = 0.5*0.95
            # pred = np.random.choice([1, 0], 1, p=[1-pp, pp])[0]
            return [[1, 1, 0, listidstatus], [1, 1, 1, listidstatus]]
        else:
            return [[1, 1, pred, listidstatus]]
    elif a == 1:
        ''' PlaceOnConveyor '''
        ''' After we attempt to place on conveyor, pred should become unknown '''
        return [[4, 0, 2, listidstatus]]
    elif a == 2:
        ''' PlaceInBin '''
        if listidstatus == 1:
            # pp = 1-(2/numObjects)
            # listidstatus = np.random.choice([1, 0], 1, p=[pp, 1-pp])[0]
            ''' After we attempt to place in bin, pred should become unknown '''
            return [[2, 2, 2, 0], [2,  2, 2, 1]]
        else:
            return [[2, 2, 2, listidstatus]]
    elif a == 3:
        ''' Pick '''
        return [[3, 3, pred, listidstatus]]
    elif a == 4:
        ''' ClaimNewOnion '''
        return [[0, eefLoc, 2, listidstatus]]
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
            return [[0, eefLoc, 2, 2]]
    return


def isValidState(onionLoc, eefLoc, pred, listidstatus, s, ns):
    #  
    if (onionLoc == 0 and eefLoc == 1) or (onionLoc == 1 and eefLoc != 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3) or (onionLoc == 4 and eefLoc != 0) or s == ns:
        return False
    return True


def isValidNxtState(a, onionLoc, eefLoc, pred, listidstatus):
    if (onionLoc == 1 and eefLoc != 1) or (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3) or (onionLoc == 4 and eefLoc != 0):
        return False
    if a == 1 or a == 2:
        if (eefLoc == 0 and pred != 2) or (eefLoc == 2 and pred != 2):
            return False
    return True


def getKeyFromValue(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"