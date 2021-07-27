import numpy as np
import matplotlib.pyplot as plt

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Controllers.grid_rl_controller import GridRLController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.grid_rl_policy import Grid_RL_Policy
from Utils.utils import generate_episode
from Logger.logger import Logger


'''Environment Parameters'''
numrobot    = 6
gridwidth   = 25
gridlen     = 25
seed        = 420
num_actions = 4
numsteps    = 50
render      = True
lr          = 0.01

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
# policy = Basic_Random(numrobot, action_space)
policy = Grid_RL_Policy(numrobot, action_space, lr)

'''Making the Controller for the Swarm Agent'''
# controller = GridRLRandomController(numrobot, policy)
controller = GridRLController(numrobot, policy)

'''Making the Environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, seed=seed)

#logging parameters
makevid = False
testname = "grid_rl"
logger = Logger(testname, makevid, 0.02)

# test generating an episode
episode, steps = generate_episode(env, controller, iters=10)

#tracking rewards
rewardlis = []

# main loop
state = env.reset()
for i in range(numsteps):
    if (i + 1) % 5 == 0:
        print("-----------------iteration: " + str(i + 1) + "-----------------")

    # get action and advance environment
    action = controller.getControls(state)
    state, reward = env.step(action)

    # track reward
    rewardlis.append(reward)

    # render agents if necessary
    if render:
        env.render()
        logger.update()

#closing logger
logger.close()

# plot rewards
plt.figure(2)
plt.title("Reward per Iteration")
plt.xlabel('Iterations')
plt.ylabel('Reward')
line_r, = plt.plot(rewardlis, label="Reward")
plt.legend(handles=[line_r])
plt.show()
