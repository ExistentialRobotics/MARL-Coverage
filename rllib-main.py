import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from super_grid_rl import SuperGridRL

ray.init()

# tune.run(PPOTrainer,
#          stop={
#              # "episode_reward_mean": 200,
#              "timesteps_total": 100000
#          },
#          config={
#              "env": SuperGridRL,
#              "rollout_fragment_length" : 100,
#              "num_sgd_iter" : 30,
#              "train_batch_size" : 4000,
#              "num_gpus": 1,
#              "env_config": {
#                  "numrobot" : 3,
#                  "gridlen"  : 25,
#                  "gridwidth"  : 25,
#                  "maxsteps"  : 100,
#                  "discrete_grid_values"  : 2,
#                  "collision_penalty"  : 5,
#                  "sensesize"  : 1,
#                  "grid"  : None,
#                  "seed"  : 1729,
#                  "free_penalty"  : 0,
#                  "use_scanning"  : True,
#                  "p_obs"  : 0
#                  },
#              "model" : {
#                  "dim" : 25,
#                  "conv_filters" :
#                  [[64, [5, 5], 2],
#                  [32, [5, 5], 2],
#                  [512, [7, 7], 1]],
#                  "conv_activation": "relu",
#                  "post_fcnet_hiddens": [512, 100],
#                  "post_fcnet_activation": "relu"
#                  },
#              "framework" : "torch"
#              # "lr": tune.grid_search([0.01, 0.001, 0.0001])
#          })


#rainbow
# tune.run(DQNTrainer,
#          stop={
#              "episode_reward_mean": 600,
#              "timesteps_total": 10000000
#          },
#          config={
#              "env": SuperGridRL,
#              "num_gpus": 1,
#              "env_config": {
#                  "numrobot" : 3,
#                  "gridlen"  : 25,
#                  "gridwidth"  : 25,
#                  "maxsteps"  : 100,
#                  "discrete_grid_values"  : 2,
#                  "collision_penalty"  : 5,
#                  "sensesize"  : 1,
#                  "grid"  : None,
#                  "seed"  : 1729,
#                  "free_penalty"  : 0,
#                  "use_scanning"  : True,
#                  "p_obs"  : 0
#                  },
#              "gamma" : 0.85,
#             "exploration_config": {
#                 # The Exploration class to use.
#                 "type": "EpsilonGreedy",
#                 # Config for the Exploration class' constructor:
#                 "initial_epsilon": 1.0,
#                 "final_epsilon": 0.0,
#                 "epsilon_timesteps": 2,  # Timesteps over which to anneal epsilon.
#             },
#              "hiddens" : [512, 128],
#              "model" : {
#                  "conv_filters" :
#                  [[64, [5, 5], 2],
#                   [32, [5, 5], 2],
#                  [512, [7, 7], 1]],
#                  # "conv_activation": "relu",
#                  # "post_fcnet_hiddens": [512, 100],
#                  # "post_fcnet_activation": "relu"
#                  },
#              "framework" : "torch",
#              # "lr": tune.grid_search([0.01, 0.001, 0.0001])
#              "lr" : 0.0001,
#              "train_batch_size" : 250,
#              "rollout_fragment_length": 100,
#              "num_workers" : 2,
#              "horizon" : 100,
#              "no_done_at_end" : True,

#              # "training_intensity": None

#              #rainbow related configs
#              "learning_starts" : 10000,
#              "buffer_size" : 50000,
#              "num_atoms" : 51,
#              "noisy" : True,
#              "n_step" : 3


#          })



tune.run(DQNTrainer,
         stop={
             "episode_reward_mean": 600,
             "timesteps_total": 10000000
         },
         config={
             "env": SuperGridRL,
             "num_gpus": 1,
             "env_config": {
                 "numrobot" : 3,
                 "gridlen"  : 25,
                 "gridwidth"  : 25,
                 "maxsteps"  : 100,
                 "discrete_grid_values"  : 2,
                 "collision_penalty"  : 5,
                 "sensesize"  : 1,
                 "grid"  : None,
                 "seed"  : 1729,
                 "free_penalty"  : 0,
                 "use_scanning"  : True,
                 "p_obs"  : 0
                 },
             "gamma" : 0.85,
            "exploration_config": {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 300000,  # Timesteps over which to anneal epsilon.
            },
             "hiddens" : [512, 128],
             "model" : {
                 "conv_filters" :
                 [[64, [5, 5], 2],
                  [32, [5, 5], 2],
                 [512, [7, 7], 1]],
                 # "conv_activation": "relu",
                 # "post_fcnet_hiddens": [512, 100],
                 # "post_fcnet_activation": "relu"
                 },
             "framework" : "torch",
             # "lr": tune.grid_search([0.01, 0.001, 0.0001])
             # "lr" : 0.0001,
             "train_batch_size" : 250,
             "rollout_fragment_length": 50,
             "num_workers" : 2,
             "horizon" : 100,
             "no_done_at_end" : True,

             # "training_intensity": None

             #better buffer???
             # "prioritized_replay_alpha": 0.5,
             # "final_prioritized_replay_beta": 1.0,
             # "prioritized_replay_beta_annealing_timesteps": 400000

         })

ray.shutdown()
