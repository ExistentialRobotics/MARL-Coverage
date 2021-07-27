import numpy as np
import matplotlib.pyplot as plt

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Controllers.grid_rl_controller import GridRLController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.policy_gradient import PolicyGradient
from Logger.logger import Logger
from Utils.utils import train_RLalg

DASH = "-----------------------------------------------------------------------"

'''Environment Parameters'''
numrobot       = 6
gridwidth      = 25
gridlen        = 25
seed           = 420
num_actions    = 4
render         = True
lr             = 0.01
train_episodes = 100
test_episodes  = 100
iters          = 100

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
policy = PolicyGradient(numrobot, action_space, lr)

'''Making the Controller for the Swarm Agent'''
controller = GridRLController(numrobot, policy)

'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, seed=seed)

#logging parameters
makevid = False
testname = "grid_rl"
logger = Logger(testname, makevid, 0.02)

'''Train policy'''
print("----------Running PG for " + str(test_episodes) + " episodes-----------")
train_rewardlis = train_RLalg(env, controller, episodes=train_episodes, iters=iters)

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
test_rewardlis = []
success = 0
for _ in range(test_episodes):
    if _ % 10 == 0:
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
    test_rewardlis.append(total_reward)

'''Display results'''
print(DASH)
print("Trained policy successfully covered the environment " + str((success / test_episodes) * 100) + " percent of the time!")
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
