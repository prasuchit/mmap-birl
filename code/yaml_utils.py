import yaml
import numpy as np
import os
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
        self.InitialStates = np.round(mdp.start[0], 6)
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
        filename = self.name+'_mdp.yaml'
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as f:
            f.write('%YAML 1.1\n')
            f.write('---\n')
            f.write('NumStates: '+str(self.NumStates)+'\n')
            f.write('NumActions: '+str(self.NumActions)+'\n')
            f.write('Gamma: '+str(self.Gamma)+'\n')
            f.write('Transitions:\n')
            for i in range(self.NumStates):
                for j in range(self.NumActions):
                    f.write('  ? ['+str(i)+', '+str(j)+']\n')
                    f.write('  : [')
                    for k in range(self.NumStates):
                        if k < (self.NumStates - 1):
                            f.write(str(self.Transitions[i, j, k])+', ')
                        else:
                            f.write(str(self.Transitions[i, j, k]))
                    f.write(']\n')

            f.write('InitialStates: [')
            for i in range(len(self.InitialStates)):
                if i < len(self.InitialStates - 1):
                    if self.name == 'sorting':
                        f.write(str(self.InitialStates[i][0])+',')
                    else:
                        f.write(str(self.InitialStates[i])+',')
                else:
                    if self.name == 'sorting':
                        f.write(str(self.InitialStates[i][0]))
                    else:
                        f.write(str(self.InitialStates[i][0]))
            f.write(']\n')
            f.write('RewardFeatures:\n')
            for i in range(self.NumStates):
                for j in range(self.NumActions):
                    f.write('  ? ['+str(i)+', '+str(j)+']\n')
                    f.write('  : [')
                    for k in range(self.nF):
                        if k < (self.nF - 1):
                            f.write(str(self.RewardFeatures[i, j][k])+',')
                        else:
                            f.write(str(self.RewardFeatures[i, j][k]))
                    f.write(']\n')

        filename = 'trajs.yaml'
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as f:
            f.write('%YAML 1.1\n')
            f.write('---\n')
            for i in range(self.nTrajs):
                f.write('Trajectory_'+str(i)+':\n')
                for j in range(self.nSteps):
                    f.write('  - ['+str(self.trajs[i, j][0]) +
                            ', '+str(self.trajs[i, j][1])+']\n')
        print("Hey")
