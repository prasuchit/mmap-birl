import yaml
import numpy as np
import os
import utils3

'''
NOTE: This module is still under construction. Essentially, this loads in 
a domain and trajectories yaml file and populates the MDP details and trajectories
to work with MMAP-BIRL algorithm. Raise an issue or email me at ps32611@uga.edu if 
you want me to implement this soon.
'''
class YAMLGenerator():
    def __init__(self, mdp = None, expertData = None) -> None:
        self.name = mdp.name
        self.NumStates = mdp.nStates
        self.NumActions = mdp.nActions
        self.NumObservations = self.NumStates+1
        self.Gamma = mdp.discount
        self.Transitions = np.transpose(np.round(mdp.transition, 6), (1, 2, 0)) # Making it (s,a,s') from (s',s,a)
        self.InitialStates = np.round(mdp.start, 6)
        self.RewardFeatures = np.zeros(
            (self.NumStates, self.NumActions, mdp.nFeatures), dtype=int)
        self.nF = mdp.nFeatures
        self.nTrajs = expertData.nTrajs
        self.nSteps = expertData.nSteps
        self.Trajectories = expertData.trajSet
        for i in range(self.NumActions):
            for j in range(self.NumStates):
                self.RewardFeatures[j, i, :] = mdp.F[j, :]

        self.Observations = np.zeros((self.NumStates, self.NumActions, self.NumStates + 1)) # num of obsv = (nS*nA + 1); 1 is for the occlusion, which is also an obsv.
        self.generateObsvFunc()

    def generateObsvFunc(self):
        
        p_o = 1/((self.NumStates*self.NumActions) + 1)  # Prob of occl = Uniform dist of 1/(num of obsv)
        for s in range(self.NumStates):
            for a in range(self.NumActions):
                if self.name == 'sorting':
                    onionLoc, eefLoc, pred = utils3.sid2vals(s)
                    s_noisy = utils3.vals2sid(onionLoc, eefLoc, int(not pred))
                    if pred != 2:
                        pp = 0.3*0.95
                        self.Observations[s,a,s_noisy] = pp
                        self.Observations[s,a,s] = 1 - pp - p_o
                    else:
                        self.Observations[s,a,s] = 1 - p_o

                elif self.name == 'gridworld':
                    if s == 15:
                        pp = 0.0
                        self.Observations[s,a,14] = pp # 20% chance that we got state 14 instead.
                        self.Observations[s,a,s] = 1 - pp - p_o 
                    else:
                        self.Observations[s,a,s] = 1 - p_o

                self.Observations[s, a, self.NumStates] = p_o

        # Check obsv probability
        for a in range(self.NumActions):
            for s in range(self.NumStates):
                err = abs(sum(self.Observations[s, a, :]) - 1)
                if err > 1e-6 or np.any(self.Observations) > 1 or np.any(self.Observations) < 0:
                    print(f"self.Observations({s},{a}, :) = {self.Observations[s, a, :]}")
                    print('ERROR: \n', s, a, np.sum(self.Observations[s, a, :]))
                                   

    def writeVals(self):
        filename = self.name+'_mdp.yaml'
        if os.path.exists(filename):
            os.remove(filename)

        data_dump = {'NumStates': self.NumStates, 'NumActions': self.NumActions, 'NumObservations': self.NumObservations, 'Gamma': self.Gamma, 'nF': self.nF, 'nTrajs': self.nTrajs, 'nSteps': self.nSteps, 'Transitions': self.Transitions.tolist(), 
        'Observations': self.Observations.tolist(), 'InitialStates': self.InitialStates.tolist(), 'RewardFeatures': self.RewardFeatures.tolist(), 'Trajectories': self.Trajectories}
        
        with open(filename, 'w') as f:
            yaml.dump(data_dump, f, default_flow_style=False)


    def readVals(self):
        with open(self.name+'_mdp.yaml') as f:
            data_loaded = yaml.load(f, Loader=yaml.Loader)
        
            self.Gamma = float(data_loaded['Gamma'])
            self.InitialStates = np.array(data_loaded['InitialStates'])
            self.nF = int(data_loaded['nF'])
            self.nSteps = int(data_loaded['nSteps'])
            self.nTrajs = int(data_loaded['nTrajs'])
            self.NumActions = int(data_loaded['NumActions'])
            self.NumObservations = int(data_loaded['NumObservations'])
            self.NumStates = int(data_loaded['NumStates'])
            self.Observations = np.array(data_loaded['Observations'])
            self.RewardFeatures = np.array(data_loaded['RewardFeatures'])
            self.Transitions = np.array(data_loaded['Transitions'])
            self.Trajectories = np.array(data_loaded['Trajectories'])
            print('Complete')

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
        #             
        #     f.write('Observations:\n')
        #     for i in range(self.NumStates):
        #         for j in range(self.NumActions):
        #             f.write('  ? ['+str(i)+', '+str(j)+']\n')
        #             f.write('  : [')
        #             for k in range(self.NumStates + 1):
        #                 if k < (self.NumStates):
        #                     f.write(str(self.Observations[i, j][k])+',')
        #                 else:
        #                     f.write(str(self.Observations[i, j][k]))
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

        # filename = 'trajs.yaml'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # with open(filename, 'w') as f:
        #     f.write('%YAML 1.1\n')
        #     f.write('---\n')
        #     for i in range(self.nTrajs):
        #         f.write('Trajectory_'+str(i)+':\n')
        #         for j in range(self.nSteps):
        #             if self.trajs[i, j][0] == -1:
        #                 f.write('  - ['+str(self.NumStates)+']\n')
        #             else:
        #                 f.write('  - ['+str(self.trajs[i, j][0])+']\n')
        # print("Hey")
