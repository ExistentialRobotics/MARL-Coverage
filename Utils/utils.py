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
        if controller._replay_buffer is not None:
            controller._replay_buffer.addtransition(state, action, reward, next_state)

        state = next_state
        steps += 1
        total_reward += reward
        done = env.done()

        if render:
            env.render()

    return episode, total_reward, steps

def train_RLalg(env, controller, logger, episodes=1000, iters=100, use_buf=False, render=False):
    h = episodes // 2

    # reset environment
    state = env.reset()

    # set policy network to train mode
    controller.set_train()

    reward_per_episode = []
    best_reward = -sys.maxsize - 1
    for _ in range(episodes):
        r = False
        if _ == h:
            if render:
                r = True
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        episode, total_reward, steps = generate_episode(env, controller, iters=iters, render=r)

        # sample transitions from replay buffer to update the policy
        if use_buf:
            episode = controller._replay_buffer.sampleepisode(steps=steps)

        # track reward per episode
        reward_per_episode.append(total_reward)

        # letting us know when we beat previous best
        if total_reward > best_reward:
            print("New best reward on episode " + str(_) + ": " + str(total_reward))
            best_reward = total_reward


        #saving policy at fixed checkpoints
        if _ % 500 == 0:
            logger.saveModelWeights(controller._policy.policy_net)

        # update policy using the episode
        controller.update_policy(episode)

    #saving final policy
    logger.saveModelWeights(controller._policy.policy_net)

    return reward_per_episode
