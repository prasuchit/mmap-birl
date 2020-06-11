This file creates the necessary parameters needed to represent a gridworld as an MDP. It has two functions:

1. Init: Uses the mdp class from models.py and assigns values. 
Finds the possible next states the agent/expert could land in using the loc2s function and assigns transition probability 
to the intended next state and the others based on the amount of noise in the system. 
Also creates a feature vector for all the states based on which feature they fall under. 
Assigns start state probability to each state (defines how likely the agent/expert is to start from that state).

2. loc2s: Given the x,y location of the agent/expert, returns the corresponding state number.
