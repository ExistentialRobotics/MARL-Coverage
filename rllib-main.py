import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from super_grid_rl import SuperGridRL

ray.init()
tune.run(PPOTrainer,
         stop={
             # "episode_reward_mean": 200,
             "timesteps_total": 100000
         },
         config={
             "env": SuperGridRL,
             "rollout_fragment_length" : 100,
             "num_sgd_iter" : 30,
             "train_batch_size" : 4000,
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
             "model" : {
                 "dim" : 25,
                 "conv_filters" :
                 [[64, [5, 5], 2],
                 [32, [5, 5], 2],
                 [512, [7, 7], 1]],
                 "conv_activation": "relu",
                 "post_fcnet_hiddens": [512, 100],
                 "post_fcnet_activation": "relu"
                 },
             "framework" : "torch"
             # "lr": tune.grid_search([0.01, 0.001, 0.0001])
         })

ray.shutdown()
