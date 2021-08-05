import numpy as np
import matplotlib.pyplot as plt
import sys
import json

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Controllers.grid_rl_controller import GridRLController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.policy_gradient import PolicyGradient
from Policies.dqn import DQN
from Policies.replaybuffer import ReplayBuffer
from Logger.logger import Logger
from Utils.utils import train_RLalg
import torch.nn as nn

DASH = "-----------------------------------------------------------------------"

# prevent decimal printing
np.set_printoptions(suppress=True)

'''Read config file'''
if len(sys.argv) != 2:
    print(DASH)
    print("No config file specified.")
    print(DASH)
    sys.exit(1)

# check if config path is valid
config_path = sys.argv[1]
try:
    config_file = open(config_path)
except OSError:
    print(DASH)
    print(str(config_path) + " does not exit.")
    print(DASH)
    sys.exit(1)

# load json file
try:
    exp_parameters = json.load(config_file)
except:
    print(DASH)
    print(str(config_path) + " is an invalid json file.")
    print(DASH)
    sys.exit(1)

'''Environment Parameters'''
exp_name       = exp_parameters["experiment_name"]
numrobot       = exp_parameters["numrobot"]
gridwidth      = exp_parameters["gridwidth"]
gridlen        = exp_parameters["gridlen"]
seed           = exp_parameters["seed"]
num_actions    = exp_parameters["num_actions"]
lr             = exp_parameters["lr"]
train_episodes = exp_parameters["train_episodes"]
test_episodes  = exp_parameters["test_episodes"]
train_iters    = exp_parameters["train_iters"]
test_iters     = exp_parameters["test_iters"]
collision_p    = exp_parameters["collision_p"]
buffer_maxsize = (train_episodes * train_iters) / exp_parameters["buf_divisor"]

makevid = False
if exp_parameters["makevid"] == 1:
    makevid = True


render_test = False
if exp_parameters["render_test"] == 1:
    render_test = True

render_train = False
if exp_parameters["render_train"] == 1:
    render_train = True

buffer = False
if exp_parameters["buffer"] == 1:
    buffer = True

conv_channels = []
for channel in exp_parameters["conv_channels"]:
    conv_channels.append(channel["_"])

conv_filters = []
for filter in exp_parameters["conv_filters"]:
    conv_filters.append((filter["_"], filter["_"]))

hidden_sizes = []
for size in exp_parameters["hidden_sizes"]:
    hidden_sizes.append(size["_"])

if exp_parameters["conv_activation"] == "relu":
    conv_activation = nn.ReLU

if exp_parameters["hidden_activation"] == "relu":
    hidden_activation = nn.ReLU

if exp_parameters["output_activation"] == "sigmoid":
    output_activation = nn.Sigmoid

print(DASH)
print("Running experiement using: " + str(config_path))
print(DASH)

'''Init logger'''
logger = Logger(exp_name, makevid, 0.02)

'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, collision_penalty=collision_p)

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
obs_dim = np.squeeze(env.get_state(), axis=0).shape
random_policy = False
if exp_parameters["random_policy"] == "random":
    policy = Basic_Random(numrobot, action_space)
    random_policy = True
elif exp_parameters["random_policy"] == "pg":
    policy = PolicyGradient(numrobot, action_space, lr, obs_dim, conv_channels,
                            conv_filters, conv_activation, hidden_sizes,
                            hidden_activation, output_activation)
elif exp_parameters["random_policy"] == "dqn":
    #TODO add other DQN parameters into config
    policy = DQN(numrobot, action_space, lr, obs_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation, output_activation)

# '''Init replay buffer'''
# buff = None
# if buffer:
#     buff = ReplayBuffer(buffer_maxsize)

'''Making the Controller for the Swarm Agent'''
controller = GridRLController(numrobot, policy)

'''Train policy'''
train_rewardlis = []
if not random_policy:
    print("----------Running PG for " + str(train_episodes) + " episodes-----------")
    train_rewardlis = train_RLalg(env, controller, logger, episodes=train_episodes, iters=train_iters, render=render_train)
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

# plot testing rewards
plt.figure(2)
plt.title("Training Reward per Episode")
plt.xlabel('Episodes')
plt.ylabel('Reward')
line_r, = plt.plot(train_rewardlis, label="Training Reward")
plt.legend(handles=[line_r])
logger.savefig(plt.gcf(), 'TrainingReward')
plt.show()

# plot training rewards
plt.figure(3)
plt.title("Testing Reward per Episode")
plt.xlabel('Episodes')
plt.ylabel('Reward')
line_r, = plt.plot(test_rewardlis, label="Testing Reward")
plt.legend(handles=[line_r])
logger.savefig(plt.gcf(), 'TestingReward')
plt.show()

#closing logger
logger.close()
