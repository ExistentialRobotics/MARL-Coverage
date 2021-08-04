import numpy as np
import matplotlib.pyplot as plt

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Controllers.grid_rl_controller import GridRLController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.policy_gradient import PolicyGradient
from Policies.replaybuffer import ReplayBuffer
from Logger.logger import Logger
from Utils.utils import train_RLalg
import torch.nn as nn

DASH = "-----------------------------------------------------------------------"

'''Environment Parameters'''
numrobot          = 6
gridwidth         = 25
gridlen           = 25
seed              = 420
num_actions       = 4
render_test       = True
render_train      = False
lr                = 0.0001
train_episodes    = 200
test_episodes     = 100
iters             = 100
collision_p       = 5
conv_channels     = [10, 10]
conv_filters      = [(5, 5), (5, 5)]
conv_activation   = nn.ReLU
hidden_sizes      = [500, 100]
hidden_activation = nn.ReLU
output_activation = nn.Sigmoid
buffer = True
buffer_maxsize = 500

'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, collision_penalty=collision_p)

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
obs_dim = np.squeeze(env.get_state(), axis=0).shape
policy = PolicyGradient(numrobot, action_space, lr, obs_dim, conv_channels,
                        conv_filters, conv_activation, hidden_sizes,
                        hidden_activation, output_activation)

'''Init replay buffer'''
buff = None
if buffer:
    buff = ReplayBuffer(buffer_maxsize)

'''Making the Controller for the Swarm Agent'''
controller = GridRLController(numrobot, policy, replay_buffer=buff)

#logging parameters
makevid = False
testname = "grid_rl"
logger = Logger(testname, makevid, 0.02)

'''Train policy'''
print("----------Running PG for " + str(test_episodes) + " episodes-----------")
train_rewardlis = train_RLalg(env, controller, episodes=train_episodes, iters=iters)

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
# set policy network to eval mode
controller.set_eval()

test_rewardlis = []
success = 0
percent_covered = 0
for _ in range(test_episodes):
    render = False
    if _ % 10 == 0:
        if render_test:
            render = True
        print("Testing Episode: " + str(_) + " out of " + str(test_episodes))

    # reset env at the start of each episode
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False
    while not done and steps != iters:
        # determine action
        action = controller.getControls(state, testing=True)

        # step environment and save episode results
        state, reward = env.step(action)
        steps += 1
        total_reward += reward

        # render if necessary
        if render:
            env.render()
            logger.update()

        # determine if env was successfully covered
        done = env.done()
        if done:
            success += 1
    percent_covered += env.percent_covered()
    test_rewardlis.append(total_reward)

'''Display results'''
print(DASH)
print("Trained policy covered " + str((percent_covered / test_episodes) * 100) + " percent of the environment on average!")
print(DASH)

#closing logger
logger.close()

# plot testing rewards
plt.figure(2)
plt.title("Training Reward per Episode")
plt.xlabel('Episodes')
plt.ylabel('Reward')
line_r, = plt.plot(train_rewardlis, label="Training Reward")
plt.legend(handles=[line_r])
plt.show()

# plot training rewards
plt.figure(3)
plt.title("Testing Reward per Episode")
plt.xlabel('Episodes')
plt.ylabel('Reward')
line_r, = plt.plot(test_rewardlis, label="Testing Reward")
plt.legend(handles=[line_r])
plt.show()
