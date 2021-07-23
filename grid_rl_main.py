import numpy as np
import matplotlib.pyplot as plt

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random


'''Environment Parameters'''
numrobot    = 6
gridwidth   = 25
gridlen     = 25
seed        = 420
num_output  = 6
num_actions = 4
numsteps    = 20
render      = True

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
policy = Basic_Random(num_output, action_space)

'''Making the Controller for the Swarm Agent'''
c = GridRLRandomController(numrobot, policy)

'''Making the Environment'''
e = SuperGridRL(c, numrobot, gridlen, gridwidth, seed=seed)


#tracking rewards
rewardlis = []

# main loop
for i in range(numsteps):
    if (i + 1) % 5 == 0:
        print("-----------------iteration: " + str(i + 1) + "-----------------")
    r = e.step()
    rewardlis.append(r)
    if render:
        e.render()

# plot rewards
plt.figure(2)
plt.title("Reward per Iteration")
plt.xlabel('Iterations')
plt.ylabel('Reward')
line_r, = plt.plot(rewardlis, label="Reward")
plt.legend(handles=[line_r])
plt.show()
