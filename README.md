# MMAP-BIRL

This project builds on the foundation established by J. Choi and K. Kim in MAP inference for Bayesian inverse reinforcement learning, NIPS 2010.

The algorithm developed in this repo considers occluded and noisy expert demonstrations and tries to learn the best reward function using Marginalized Maximum a Posteriori method with Bayesian IRL inference technique.

The codebase has been built and developed in python based on an existing repo by Jaedeug Choi (jdchoi@ai.kaist.ac.kr) in Matlab. 

This paper has been published in UAI 2022 conference. The PDF is currently available at this [link](https://proceedings.mlr.press/v180/suresh22a/suresh22a.pdf). 

Cite this work as:

@inproceedings{suresh2022marginal,
  title={Marginal MAP estimation for inverse RL under occlusion with observer noise},
  author={Suresh, Prasanth Sengadu and Doshi, Prashant},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1907--1916},
  year={2022},
  organization={PMLR}
}

## Requirements: 

I'd recommend creating a conda env and installing the following packages within:

    1. numpy    
    2. scipy    
    3. logging    
    4. pymdptoolbox    
    5. tqdm    
    6. pyyaml    
    7. multiprocessing

I will soon push a requirements.txt file that you can use to create the environment easily.

## Usage:

Assuming you've read the paper and have a good understanding of how the algorithm works, first open /yaml_files/init_params.yaml and make sure the name of the problem you want to test and all other parameters are to your requirements. If you're unsure about the hyperparameters or want to just test an existing domain, you could just run the code as it is and it would work fine.

Activate your conda env and cd into mmap-birl/code folder and run the following command:

   `python3 runner.py`

You should see the results printed on the terminal that tells you about the initial weights sampled, learned reward weights and the metrics - Inverse Learning Error(ILE)/Value Difference, Learned Behavior Accuracy(LBA)/Policy Difference and Reward Difference.

### Pending Updates:

Ideally, I plan to make this such that you can pass a yaml file for the MDP and a yaml file with trajectories and execute MMAP-BIRL. That functionality is under works.

The sparse MDP implementation mostly works, but I haven't done extensive testing with it yet.

The scipy minimize based optimization is deprecated for now, although I plan to bring it back soon.

Parallel processing is available for the marginalization part, although for small mdps and/or almost deterministic transitions, it won't matter much.

The observation function and s-a trajectories sampled from it needs to be more robust, but for the current domains, it works fine.

The domains could be made much more stochastic in terms of transition probabilities. That would be an added challenge for MMAP-BIRL to converge.

### Feel free to raise issues, I'll address them as soon as I can.
