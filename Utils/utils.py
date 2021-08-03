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
        episode.append((state, action, reward))

        state = next_state
        steps += 1
        total_reward += reward
        done = env.done()

        if render:
            env.render()

    return episode, total_reward, steps

def train_RLalg(env, controller, episodes=1000, iters=100, render=False):
    # reset environment
    state = env.reset()

    # set policy network to train mode 
    controller.set_train()

    reward_per_episode = []
    best_reward = -sys.maxsize - 1
    for _ in range(episodes):
        r = False
        if _ % 10 == 0:
            if render:
                r = True
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        # generate episode
        episode, total_reward, steps = generate_episode(env, controller, iters=iters, render=r)

        # track reward per episode
        reward_per_episode.append(total_reward)

        # save the best policy
        #TODO fix this, reward is super noisy, better metric might be a moving average, or just save at a fixed interval
        if total_reward > best_reward:
            # print("New best reward on episode " + str(_) + ": " + str(total_reward) + "! Saving policy!")
            best_reward = total_reward
            # controller.save_policy()

        # update policy using the episode
        controller.update_policy(episode)

    #saving final policy
    controller.save_policy()

    return reward_per_episode
