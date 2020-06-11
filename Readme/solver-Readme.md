This file has two functions: 

1. Policy Iteration: This is the regular policy iteration method used to solve an MDP, thus, here an [MDP Toolbox](https://pymdptoolbox.readthedocs.io/en/latest/) has been used
to solve for the desired results. Q is obtained from V using a function in utils.py
2. Solver: To solve for H, which is used to compute V (in the naive method) and also check for optimal region for gradient reuse. Equations for this are provided here: [1](https://papers.nips.cc/paper/4479-map-inference-for-bayesian-inverse-reinforcement-learning),
[2](https://papers.nips.cc/paper/4479-map-inference-for-bayesian-inverse-reinforcement-learning-supplemental.zip)
