import numpy as np
import sys
import time

def generate_episode(env, policy, logger, render=False, makevid=False, ignore_done=True, testing=False, ind=None, phase_1=None):
    # reset env at the start of each episode
    episode = []
    state, grid = env.reset(testing, ind)
    total_reward = 0
    done = False

    #reset policy at beginning of episode
    policy.reset(testing)

    # iterate till episode completion
    i = 0
    while not done:
        # determine action
        policy.set_reward(total_reward)
        action = policy.pi(state, phase_1=phase_1)

        # step environment and save episode results
        next_state, reward = env.step(action)
        policy.add_state(str(next_state[0]))

        # determine if episode is completed
        done = env.done()

        if i % 50 == 0:
            print(i)
        i += 1

        if testing and env._currstep == env._test_maxsteps:
            done = True
        elif not testing and env._currstep == env._train_maxsteps:
            done = True

        #adding variables to episode
        episode.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)

    return episode, total_reward

def train_RLalg(env, policy, logger, train_episodes=1000, test_episodes=10, render=False,
                checkpoint_interval=500, ignore_done=True):
    p_1_thres = train_episodes / 3

    # set policy network to train mode
    policy.set_train()

    # list for statistics
    reward_per_episode = []
    losslist = []
    test_percent_covered = []

    best_reward = -sys.maxsize - 1
    checkpoint_num = 0
    for _ in range(train_episodes):
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(train_episodes) + " num nodes in tree: " + str(len(policy._nodes)))

        # obtain the next episode
        if p_1_thres > _:
            phase_1 = True
        else:
            phase_1 = False
        episode, total_reward = generate_episode(env, policy, logger, render=False, makevid=False, ignore_done=ignore_done, phase_1=phase_1)

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
            testrewards, average_percent_covered = test_RLalg(env, policy, logger, episodes=test_episodes, render_test=render)
            test_percent_covered.append(average_percent_covered)
            policy.set_train()

            #printing debug info
            checkpoint_num += 1
            print("Checkpoint Policy {} covered ".format(checkpoint_num) + str(average_percent_covered) + " percent of the environment on average!")

        # update policy using the episode
        policy.update_policy(episode)

        #tracking training loss for the episode
        losslist.append(policy._avgloss)

    #saving final policy
    logger.saveModelWeights(policy.getnet())

    return reward_per_episode, losslist, test_percent_covered


def test_RLalg(env, policy, logger, episodes=100, render_test=False, makevid=False):
    # set model to eval mode
    policy.set_eval()
    if env._test_gridlis is not None:
        num_test = len(env._test_gridlis)
    else:
        num_test = 1

    test_rewardlis = []
    percent_covered = 0
    i = -1
    for _ in range(episodes * num_test):
        render = False
        if _ % 10 == 0:
            # determine if rendering the current episode
            if render_test:
                render = True
            print("Testing Episode: " + str(_) + " out of " + str(episodes * num_test))

        # determine which tesing env to use
        if _ % episodes == 0:
            i += 1

        # obtain episode
        episode, total_reward = generate_episode(env, policy, logger, render=render, makevid=makevid, testing=True, ind=i, phase_1=False)
        num_steps = len(episode)

        # track test related statistics
        percent_covered += env.percent_covered()
        test_rewardlis.append(num_steps)

    #returning policy to train mode
    policy.set_train()

    #returning the statistics
    average_percent_covered = percent_covered/(episodes * num_test)*100
    return test_rewardlis, average_percent_covered
