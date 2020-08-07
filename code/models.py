class mdp:
    # Attributes of an mdp
    def __init__(self):
        self.name = None
        self.nStates = None
        self.nActions = None
        self.nFeatures = None
        self.discount = None
        self.start = None
        self.transition = None
        self.F = None
        self.weight = None
        self.reward = None
        self.nOccs = None
        self.piL = None
        self.VL = None
        self.QL = None
        self.H = None
        self.rewardS = None
        self.transitionS = None
        self.sampled = None
