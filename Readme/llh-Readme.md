This file is where the BIRL inference happens. In the sense that, this file returns the negative log posterior and negative gradient (for scipy minimize, it returns positive values because minimizing a positive value is the same as maximizing a negative value).
It has the following functions:

1. Calc Neg Marginal Log Post: from the demonstration trajectory data, it marginalizes each occluded observation by summing up the negative 
log likelihood and gradient obtained by every state and action combination. As we know, marginalization cancels out the incorrect inputs
while summing up. As the prior is unaffected by occluded observations, the appropriate negative log prior and gradient for the given weights
is obtained and added to the negative log likelihood to get the posterior and the updated values are returned.

2. Calc Neg Log Post: This does the same as above, except without marginalizing. (Used by MAP function in birl.py)

3. Calc Log Prior: This samples the log prior for the given weights from the appropriate distribution.

4. Calc Log LLH: This calculates the gradient for the reward function using the equations from these sources: [1](https://github.com/prasuchit/mmap-irl/blob/master/mmap-irl-note.pdf),
[2](https://papers.nips.cc/paper/4479-map-inference-for-bayesian-inverse-reinforcement-learning), 
[3](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/viewPaper/6572), 
[4](https://papers.nips.cc/paper/4737-nonparametric-bayesian-inverse-reinforcement-learning-for-multiple-reward-functions),
[5](https://arxiv.org/abs/1206.5264),
[6](https://papers.nips.cc/paper/4479-map-inference-for-bayesian-inverse-reinforcement-learning-supplemental.zip)
