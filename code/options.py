import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class algorithm:
    def __init__(self, name=None, llhType=None, priorType=None):    # Parameterized constructor for class
        self.name = name
        self.llhType = llhType  # Log likelihood type
        self.priorType = priorType


class demonstrations:   
    def __init__(self): # Parameterized constructor for class
        self.weight = None
        self.policy = None
        self.trajId = None
        self.trajSet = None
        self.nTrajs = None
        self.nSteps = None
        self.trueReward = None


class trajInfo:
    def __init__(self): # Parameterized constructor for class
        self.nTrajs = None
        self.nSteps = None
        self.v = None   # State specific cumulative reward values V(s)
        self.pi = None  # Policy
        self.mu = None  # Mu(pi) is the state visitation frequency matrix for that policy pi
        self.occ = None # state occupancy (visitation frequency)
        self.featExp = None # Feature expectation
        self.cnt = None
        self.occlusions = None


class irlOptions:
    def __init__(self): # Parameterized constructor for class
        self.alg = None
        self.llhType = None # Log likelihood type
        self.priorType = None
        self.restart = None
        self.showMsg = None
        self.lb = None  # Lower bound
        self.ub = None  # Upper bound
        self.eta = None
        self.optimizer = None
        self.mu = None
        self.sigma = None

class problem:
    def __init__(self): # Parameterized constructor for class
        self.name = None
        self.iters = None
        self.discount = None
        self.nExps = None
        self.nExperts = None
        self.nTrajs = None
        self.nSteps = None
        self.initSeed = None
        self.noise = None
        self.filename = None
        self.seed = None