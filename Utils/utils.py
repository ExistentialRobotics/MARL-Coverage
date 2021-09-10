import numpy as np
import sys
import time

def generate_episode(env, policy, logger, render=False, makevid=False, ignore_done=True, testing=False):
    # reset env at the start of each episode
    episode = []
    state = env.reset()
    total_reward = 0
    done = False

    #reset policy at beginning of episode
    policy.reset()

    # iterate till episode completion
    while not done:
        # determine action
        action = policy.pi(state)

        # step environment and save episode results
        next_state, reward = env.step(action)

        # determine if episode is completed
        done = env.done()

        #checking if done happened because we ran out of time and possibly ignoring it
        new_done = done
        if ignore_done and done and env._currstep == env._maxsteps:
            new_done = False

        #adding variables to episode
        episode.append((state, action, reward, next_state, new_done))
        state = next_state
        total_reward += reward

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)

    return episode, total_reward

def train_RLalg(env, policy, logger, episodes=1000, render=False,
                checkpoint_interval=500, ignore_done=True):
    # set policy network to train mode
    policy.set_train()

    # list for statistics
    reward_per_episode = []
    losslist = []
    test_percent_covered = []

    best_reward = -sys.maxsize - 1
    checkpoint_num = 0
    for _ in range(episodes):
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        # obtain the next episode
        episode, total_reward = generate_episode(env, policy, logger, render=False, makevid=False, ignore_done=ignore_done)

        # track reward per episode
        reward_per_episode.append(total_reward)

        # letting us know when we beat previous best
        if total_reward > best_reward:
            print("New best reward on episode " + str(_) + ": " + str(total_reward))
            best_reward = total_reward

        #saving policy at fixed checkpoints and running tests
        if _ % checkpoint_interval == 0:
            #saving weights
            logger.saveModelWeights(policy.getnet())

            #testing policy
            testrewards, average_percent_covered = test_RLalg(env, policy, logger, episodes=100, render_test=render)
            test_percent_covered.append(average_percent_covered)
            policy.set_train()

            #printing debug info
            checkpoint_num += 1
            print("Checkpoint Policy {} covered ".format(checkpoint_num) + str(average_percent_covered) + " percent of the environment on average!")

        # update policy using the episode
        policy.update_policy(episode)

        #tracking training loss for the episode
        losslist.append(policy._lastloss)

    #saving final policy
    logger.saveModelWeights(policy.getnet())

    return reward_per_episode, losslist, test_percent_covered


def test_RLalg(env, policy, logger, episodes=100, render_test=False, makevid=False):
    # set model to eval mode
    policy.set_eval()

    test_rewardlis = []
    percent_covered = 0
    for _ in range(episodes):
        render = False
        if _ % 10 == 0:
            # determine if rendering the current episode
            if render_test:
                render = True
            print("Testing Episode: " + str(_) + " out of " + str(episodes))

        # obtain episode
        episode, total_reward = generate_episode(env, policy, logger, render=render, makevid=makevid, testing=True)

        # track test related statistics
        percent_covered += env.percent_covered()
        test_rewardlis.append(total_reward)

    #returning policy to train mode
    policy.set_train()

    #returning the statistics
    average_percent_covered = percent_covered/episodes*100
    return test_rewardlis, average_percent_covered
