import numpy as np
import sys

def generate_episode(env, policy, iters=100, render=False):
    # reset env at the start of each episode
    episode = []
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False

    # iterate till episode completion
    while not done and steps != iters:
        # determine action
        action = policy.step(state, False)

        # step environment and save episode results
        next_state, reward = env.step(action)
        episode.append((state, action, reward, next_state))
        state = next_state
        total_reward += np.sum(reward)

        # determine if episode is completed
        steps += 1
        done = env.done()
        if render:
            env.render()

    return episode, total_reward

def train_RLalg(env, policy, logger, episodes=1000, iters=100,  render=False,
                checkpoint_interval=500):
    # set policy network to train mode
    policy.set_train()

    reward_per_episode = []
    losslist = []
    best_reward = -sys.maxsize - 1
    checkpoint_num = 0
    for _ in range(episodes):
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        # obtain the next episode 
        episode, total_reward = generate_episode(env, policy, iters=iters, render=False)

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
            testrewards, average_percent_covered = test_RLalg(env, policy, logger, render_test=True)
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

    return reward_per_episode, losslist

def test_RLalg(env, policy, logger, episodes=10, iters=100, render_test=False,make_vid=False):
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

        # reset env at the start of each episode
        state = env.reset()
        steps = 0
        total_reward = 0
        done = False

        # iterate till episode completion
        while not done and steps != iters:
            # determine action
            action = policy.step(state, True)

            # step environment and save episode results
            state, reward = env.step(action)
            total_reward += np.sum(reward)

            # determine if episode is completed
            steps += 1
            done = env.done()

            # render if necessary
            if render:
                env.render()
                if(make_vid):
                    logger.update()

        # track test related statistics
        percent_covered += env.percent_covered()
        test_rewardlis.append(total_reward)

    #returning the statistics
    average_percent_covered = percent_covered/episodes*100
    return test_rewardlis, average_percent_covered
