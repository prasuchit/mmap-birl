This file contains the following functions:

1. Approx Eq: This is used to check if the new value is better than the old value while performing policy iteration using the naive method.
2. Sample weight: This function is called when the expert needs weights for the problem(currently gridworld).
3. Convert W2R: This function converts the weights into reward values by multiplying it with the feature vector as reward is characterized as
linear weighted combination of features in IRL.
4. Sid 2 info: UNDER CONSTRUCTION
5. Info 2 sid: UNDER CONSTRUCTION
6. Q from V: This calculates the Q matrix, (ie: the values obtained for all the actions in every state) using the V value by calculating the
expectation value and substituting in the Bellman equation.
7. Find: When provided with an array and a function, this returns values from the array that satisfy the function.
8. Get Traj Info: For the given trajectories, the state, action, visitation frequency and the occluded observations are identified and 
returned as trajectory information.
9. Sample new weight: In order to perform BIRL inference, we need to sample weights from a distribution. That is done in this funcion.
