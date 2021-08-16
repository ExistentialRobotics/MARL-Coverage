import numpy as np
import sys

def generate_episode(env, policy, logger, iters=100, render=False, makevid=False):
    # reset env at the start of each episode
    episode = []
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False

    # iterate till episode completion
    while not done and steps != iters:
        # determine action
        action = policy.pi(state)

        # step environment and save episode results
        next_state, reward = env.step(action)
        episode.append((state, action, reward, next_state))
        state = next_state
        total_reward += np.sum(reward)

        # determine if episode is completed
        steps += 1
        done = env.done()

        # render if necessary
        if render:
            env.render()
            if(makevid):
                logger.update()

    return episode, total_reward

def train_RLalg(env, policy, logger, episodes=1000, iters=100,  render=False,
                checkpoint_interval=500):
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
        episode, total_reward = generate_episode(env, policy, logger, iters=iters, render=False, makevid=False)

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
            testrewards, average_percent_covered = test_RLalg(env, policy, logger, episodes=10, render_test=render)
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

def test_RLalg(env, policy, logger, episodes=100, iters=100, render_test=False, makevid=False):
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
        episode, total_reward = generate_episode(env, policy, logger, iters=iters, render=render, makevid=makevid)

        # track test related statistics
        percent_covered += env.percent_covered()
        test_rewardlis.append(total_reward)

    #returning policy to train mode
    policy.set_train()

    #returning the statistics
    average_percent_covered = percent_covered/episodes*100
    return test_rewardlis, average_percent_covered
