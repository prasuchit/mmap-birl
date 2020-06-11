In this file we generate the data needed to implement other parts of the algorithm. This file has the following functions:

1. Generate MDP: As the name suggests, it uses the problem (currently gridworld.py) and genrerates an mdp that satisfies the input parameters.
2. Generate Demonstration: uses the demonstrations class from options.py to generate expert demonstrations to learn from. Uses sampleweight function from utils to get expert's weights for the problem. Assigns value of -1 for a random observation (state,action) to make it occluded.
3. Generate Trajectory: Calls policy iteration from solver.py, gets the trajectories, mean and varience value for all trajectories and returns them.
4. Sample Trajectories: Samples a random state from list of states and stores the state and action according to policy generated, as part of the trajectory. Repeats until length of steps. Repeats for number of trajectories. Also stores value of those states and returns the mean and varience.
