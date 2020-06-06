import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class algorithm:
    def __init__(self, name=None, llhType=None, priorType=None): 
        self.name = name
        self.llhType = llhType 
        self.priorType = priorType


class demonstrations:   
    def __init__(self):
        self.weight = None
        self.policy = None
        self.trajId = None
        self.trajSet = None
        self.nTrajs = None
        self.nSteps = None


class trajInfo:
    def __init__(self):
        self.nTrajs = None
        self.nSteps = None
        self.v = None 
        self.pi = None
        self.mu = None
        self.occ = None
        self.featExp = None
        self.cnt = None
        self.occlusions = None


class irlOptions:
    def __init__(self):
        self.alg = None
        self.llhType = None
        self.priorType = None
        self.restart = None
        self.showMsg = None
        self.lb = None
        self.ub = None
        self.eta = None
        self.optimizer = None
        self.mu = None
        self.sigma = None
        self.optiMethod = None

class problem:
    def __init__(self):
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