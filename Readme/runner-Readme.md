## This is the main file for this repo. Run the code from here.

### Main:
choice - sets the optimization method to use either scipy minimize to minimize the log posterior and gradient to arrive at the MAP reward, or,
use the manual method (for lack of a better word) to maximize the negative log posterior and gradient as described in the MAP algorithm in 
Choi and Kim's MAP paper, NIPS 2010.

The Algorithm and IRL params are set and the problem is generated using these. The corresponding MDP is generated and the expert demonstration data
is obtained. 

Using the set optimization method, the MAP reward values are obtained and compared with the expert's values based on Reward difference, 
Value difference and Policy difference.

### Compute Optimum Region: 
This computes the optimum reward region to explore next based on the current gradient and weight values.

### Reuse Cached Gradient:
Based on the reward optimality condition derived in the MAP paper, this checks if the reward lies within the optimal region and can be reused or not.

### Pi Interpretation:
Provides the corresponding action for the integer values in the policy (May be shifted into utils.py in the future).
