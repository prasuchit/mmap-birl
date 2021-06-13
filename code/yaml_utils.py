import yaml
import numpy as np
import os
import utils3

# with open(r'code\mdp.yaml') as file:
#     # The FullLoader parameter handles the conversion from YAML
#     # scalar values to Python the dictionary format
#     mdp = yaml.load(file, Loader=yaml.FullLoader)

#     print(mdp)

''' Pyyaml keeps throwing some errors while reading a working yaml file,
    so we're just going to write the output in the required format ourselves for now '''


class YAMLGenerator():
    def __init__(self, mdp=None, expertData=None) -> None:
        self.NumStates = mdp.nStates
        self.NumActions = mdp.nActions
        self.Gamma = mdp.discount
        self.Transitions = np.transpose(np.round(mdp.transition, 6), (1, 2, 0))
        self.InitialStates = np.round(mdp.start, 6)
        self.RewardFeatures = np.zeros(
            (self.NumStates, self.NumActions, mdp.nFeatures), dtype=int)
        self.nF = mdp.nFeatures
        self.nTrajs = expertData.nTrajs
        self.nSteps = expertData.nSteps
        self.trajs = expertData.trajSet
        self.name = mdp.name
        for i in range(self.NumActions):
            for j in range(self.NumStates):
                self.RewardFeatures[j, i, :] = mdp.F[j, :]

    def writeVals(self):
        # filename = self.name+'_mdp.yaml'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # with open(filename, 'w') as f:
        #     f.write('%YAML 1.1\n')
        #     f.write('---\n')
        #     f.write('NumStates: '+str(self.NumStates)+'\n')
        #     f.write('NumActions: '+str(self.NumActions)+'\n')
        #     f.write('NumObservations: '+str(self.NumStates+1)+'\n')
        #     f.write('Gamma: '+str(self.Gamma)+'\n')
        #     f.write('Transitions:\n')
        #     for i in range(self.NumStates):
        #         for j in range(self.NumActions):
        #             f.write('  ? ['+str(i)+', '+str(j)+']\n')
        #             f.write('  : [')
        #             for k in range(self.NumStates):
        #                 if k < (self.NumStates - 1):
        #                     f.write(str(self.Transitions[i, j, k])+', ')
        #                 else:
        #                     f.write(str(self.Transitions[i, j, k]))
        #             f.write(']\n')

        #     obs_prob = np.zeros((self.NumStates, self.NumActions, self.NumStates + 1))
        #     # num of obsv = (nS*nA + 1); 1 is for the occlusion, which is also an obsv.
        #     p_o = 1/((self.NumStates*self.NumActions) + 1)  # Prob of occl = Uniform dist of 1/(num of obsv)
        #     for s in range(self.NumStates):
        #         for a in range(self.NumActions):
        #             if self.name == 'sorting':
        #                 onionLoc, eefLoc, pred, listIDStatus = utils3.sid2vals(s)
        #                 s_noisy = utils3.vals2sid(onionLoc, eefLoc, int(not pred), listIDStatus)
        #                 if pred != 2:
        #                     pp = 0.3*0.95
        #                     obs_prob[s,a,s_noisy] = pp
        #                     obs_prob[s,a,s] = 1 - pp - p_o
        #                 else:
        #                     obs_prob[s,a,s] = 1 - p_o

        #             elif self.name == 'gridworld':
        #                 if s == 15:
        #                     pp = 0.0
        #                     obs_prob[s,a,14] = pp # 20% chance that we got state 14 instead.
        #                     obs_prob[s,a,s] = 1 - pp - p_o 
        #                 else:
        #                     obs_prob[s,a,s] = 1 - p_o

        #             obs_prob[s, a, self.NumStates] = p_o

        #     # Check obsv probability
        #     for a in range(self.NumActions):
        #         for s in range(self.NumStates):
        #             err = abs(sum(obs_prob[s, a, :]) - 1)
        #             if err > 1e-6 or np.any(obs_prob) > 1 or np.any(obs_prob) < 0:
        #                 print(f"obs_prob({s},{a}, :) = {obs_prob[s, a, :]}")
        #                 print('ERROR: \n', s, a, np.sum(obs_prob[s, a, :]))
            
        #     f.write('Observations:\n')
        #     for i in range(self.NumStates):
        #         for j in range(self.NumActions):
        #             f.write('  ? ['+str(i)+', '+str(j)+']\n')
        #             f.write('  : [')
        #             for k in range(self.NumStates + 1):
        #                 if k < (self.NumStates):
        #                     f.write(str(obs_prob[i, j][k])+',')
        #                 else:
        #                     f.write(str(obs_prob[i, j][k]))
        #             f.write(']\n')

        #     f.write('InitialStates: [')
        #     for i in range(len(self.InitialStates)):
        #         if i < len(self.InitialStates - 1):
        #             if self.name == 'sorting':
        #                 f.write(str(self.InitialStates[0][i][0])+',')   
        #                 ''' NOTE: There's an error here, check! '''
        #             else:
        #                 f.write(str(self.InitialStates[i][0])+',')
        #         else:
        #             if self.name == 'sorting':
        #                 f.write(str(self.InitialStates[0][i][0]))
        #                 ''' NOTE: There's an error here, check! '''
        #             else:
        #                 f.write(str(self.InitialStates[i][0]))
        #     f.write(']\n')
        #     f.write('RewardFeatures:\n')
        #     for i in range(self.NumStates):
        #         for j in range(self.NumActions):
        #             f.write('  ? ['+str(i)+', '+str(j)+']\n')
        #             f.write('  : [')
        #             for k in range(self.nF):
        #                 if k < (self.nF - 1):
        #                     f.write(str(self.RewardFeatures[i, j][k])+',')
        #                 else:
        #                     f.write(str(self.RewardFeatures[i, j][k]))
        #             f.write(']\n')

        filename = 'trajs.yaml'
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as f:
            f.write('%YAML 1.1\n')
            f.write('---\n')
            for i in range(self.nTrajs):
                f.write('Trajectory_'+str(i)+':\n')
                for j in range(self.nSteps):
                    if self.trajs[i, j][0] == -1:
                        f.write('  - ['+str(self.NumStates)+']\n')
                    else:
                        f.write('  - ['+str(self.trajs[i, j][0])+']\n')
        print("Hey")
