import numpy as np
import utils
import models
from scipy import full, sparse
import generator
import solver
import copy
import math as m
import time

def init(nOnionLoc, nPredict, nEEFLoc, discount, useSparse):
    
    # nS = 
    nA = 7
    locations = ['OnConveyor', 'InFront', 'InBin', 'Picked/AtHome', 'Placed']
    predictions = dict(zip([i for i in range(nOnionLoc)], ['Bad', 'Good', 'Unknown']))
    onionLoc = dict(zip([i for i in range(nOnionLoc)], locations[0:nOnionLoc]))
    eefLoc = dict(zip([i for i in range(nEEFLoc)], locations[0:nEEFLoc]))
    print(onionLoc, eefLoc, predictions)
    mdp = models.mdp()
    mdp.discount = discount
    return mdp


if __name__ == "__main__":
    _ = init(5,3,4,0.9,0)