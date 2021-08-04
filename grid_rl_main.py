import numpy as np
import matplotlib.pyplot as plt
import sys

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
random_policy     = False
numrobot          = 6
gridwidth         = 25
gridlen           = 25
seed              = 420
num_actions       = 4
render_test       = True
render_train      = True
lr                = 0.00001
train_episodes    = 1000
test_episodes     = 100
train_iters       = 100
test_iters        = 100
collision_p       = 5
conv_channels     = [20, 10]
conv_filters      = [(5, 5), (3, 3)]
conv_activation   = nn.ReLU
hidden_sizes      = [500, 100]
hidden_activation = nn.ReLU
output_activation = nn.Sigmoid
buffer            = True
buffer_maxsize    = (train_episodes * train_iters) / 4

# # prevent decimal printing
# np.set_printoptions(suppress=True)
#
#
# '''Read config file'''
# if len(sys.argv) != 2:
#     print(DASH)
#     print("No config file specified.")
#     print(DASH)
#     sys.exit(1)
#
# # check if config path is valid
# config_path = sys.argv[1]
# try:
#     config_file = open(config_path)
# except OSError:
#     print(DASH)
#     print(str(config_path) + " does not exit.")
#     print(DASH)
#     sys.exit(1)
#
# # load json file
# try:
#     hyperparams = json.load(config_file)
# except:
#     print(DASH)
#     print(str(config_path) + " is an invalid json file.")
#     print(DASH)
#     sys.exit(1)
#
# print(DASH)
# print("Running experiement using: " + str(config_path))
# print(DASH)


'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, collision_penalty=collision_p)

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
obs_dim = np.squeeze(env.get_state(), axis=0).shape
if random_policy:
    policy = Basic_Random(numrobot, action_space)
else:
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
train_rewardlis = []
if not random_policy:
    print("----------Running PG for " + str(test_episodes) + " episodes-----------")
    train_rewardlis = train_RLalg(env, controller, episodes=train_episodes, iters=train_iters, use_buf=buffer, render=render_train)
else:
    print("-----------------------Running Random Policy-----------------------")

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
# set policy network to eval mode
if not random_policy:
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
    while not done and steps != test_iters:
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
