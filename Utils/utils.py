import numpy as np
import sys

def generate_episode(env, controller, iters=100, render=False):
    episode = []
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False
    while not done and steps != iters:
        # determine action
        action = controller.getControls(state)

        # step environment and save episode results
        next_state, reward = env.step(action)
        episode.append((state, action, reward, next_state))
        state = next_state
        steps += 1
        total_reward += np.sum(reward)
        done = env.done()

        if render:
            env.render()

    return episode, total_reward, steps

def train_RLalg(env, controller, logger, episodes=1000, iters=100,  render=False,
                checkpoint_interval=500):
    # reset environment
    state = env.reset()

    # set policy network to train mode
    controller.set_train()

    reward_per_episode = []
    best_reward = -sys.maxsize - 1
    checkpoint_num = 0
    for _ in range(episodes):
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        episode, total_reward, steps = generate_episode(env, controller, iters=iters, render=False)

        # track reward per episode
        reward_per_episode.append(total_reward)

        # letting us know when we beat previous best
        if total_reward > best_reward:
            print("New best reward on episode " + str(_) + ": " + str(total_reward))
            best_reward = total_reward

        #saving policy at fixed checkpoints and running tests
        if _ % checkpoint_interval == 0:
            #saving weights
            logger.saveModelWeights(controller._policy.getnet())

            #testing policy
            testrewards, average_percent_covered = test_RLalg(env, controller, logger, render_test=False)

            #printing debug info
            checkpoint_num += 1
            print("Checkpoint Policy {} covered ".format(checkpoint_num) + str(average_percent_covered) + " percent of the environment on average!")

        # update policy using the episode
        controller.update_policy(episode)

    #saving final policy
    logger.saveModelWeights(controller._policy.getnet())

    return reward_per_episode

def test_RLalg(env, controller, logger, episodes=10, iters=100, render_test=False,make_vid=False):
    test_rewardlis = []
    success = 0
    percent_covered = 0
    for _ in range(episodes):
        render = False
        if _ % 10 == 0:
            if render_test:
                render = True
            print("Testing Episode: " + str(_) + " out of " + str(episodes))

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
            total_reward += np.sum(reward)

            # render if necessary
            if render:
                env.render()
                if(make_vid):
                    logger.update()

            # determine if env was successfully covered
            done = env.done()
            if done:
                success += 1
        percent_covered += env.percent_covered()
        test_rewardlis.append(total_reward)

    #returning the statistics
    average_percent_covered = percent_covered/episodes*100
    return test_rewardlis, average_percent_covered
