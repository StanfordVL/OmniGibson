# https://github.com/StanfordVL/behavior/blob/main/behavior/baselines/rl/stable_baselines3_ppo_training.py

import logging
import os
from typing import Callable

import igibson
# import behavior
from igibson.envs.igibson_env import iGibsonEnv
from igibson.wrappers import ActionPrimitiveWrapper
# from igibson.envs.skill_env import SkillEnv
from igibson import app, ig_dataset_path, example_config_path, Simulator

log = logging.getLogger(__name__)

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3_ppo_skill_example import CustomCombinedExtractor

except ModuleNotFoundError:
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
#
#         extractors = {}
#
#         total_concat_size = 0
#         feature_size = 128
#         for key, subspace in observation_space.spaces.items():
#             if key in ["proprioception", "task_obs"]:
#                 extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
#             elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
#                 n_input_channels = subspace.shape[2]  # channel last
#                 cnn = nn.Sequential(
#                     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#                     nn.ReLU(),
#                     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#                     nn.ReLU(),
#                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#                     nn.ReLU(),
#                     nn.Flatten(),
#                 )
#                 test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
#                 with th.no_grad():
#                     n_flatten = cnn(test_tensor[None]).shape[1]
#                 fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#                 extractors[key] = nn.Sequential(cnn, fc)
#             elif key in ["scan"]:
#                 n_input_channels = subspace.shape[1]  # channel last
#                 cnn = nn.Sequential(
#                     nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#                     nn.ReLU(),
#                     nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
#                     nn.ReLU(),
#                     nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
#                     nn.ReLU(),
#                     nn.Flatten(),
#                 )
#                 test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
#                 with th.no_grad():
#                     n_flatten = cnn(test_tensor[None]).shape[1]
#                 fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#                 extractors[key] = nn.Sequential(cnn, fc)
#             elif key in ['occupancy_grid']:
#                 continue
#             else:
#                 raise ValueError("Unknown observation key: %s" % key)
#             total_concat_size += feature_size
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []
#
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             if key in ['occupancy_grid']:
#                 continue
#             if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
#                 observations[key] = observations[key].permute((0, 3, 1, 2))
#             elif key in ["scan"]:
#                 observations[key] = observations[key].permute((0, 2, 1))
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         feature = th.cat(encoded_tensor_list, dim=1)
#         # print('feature.shape: ', feature.shape)  #  torch.Size([1, 384])
#         return feature


def main():
    # config_file = os.path.join('..', 'configs', "behavior_pick_and_place.yaml")
    # config_file = os.path.join('..', '..', 'configs', 'robots', "fetch_rl.yaml")
    # config_file = os.path.join(igibson.configs_path, "fetch_behavior_aps_putting_away_Halloween_decorations.yaml")
    config_file = f"{example_config_path}/behavior_mp_tiago.yaml"  # os.path.join(igibson.configs_path, "fetch_rl_cleaning_microwave_oven.yaml")
    tensorboard_log_dir = "log_dir"
    num_cpu = 1

    # def make_env(rank: int, seed: int = 0) -> Callable:
    #     def _init() -> SkillEnv:
    #         env = SkillEnv(
    #             config_file=config_file,
    #             mode="headless",
    #             action_timestep=1 / 30.0,
    #             physics_timestep=1 / 300.0,
    #             print_log=True,
    #         )
    #         env.seed(seed + rank)
    #         return env
    #
    #     set_random_seed(seed)
    #     return _init

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # env = VecMonitor(env)

    # eval_env = SkillEnv(
    #     config_file=config_file,
    #     mode="gui_interactive",
    #     # action_timestep=1 / 30.0,
    #     # physics_timestep=1 / 300.0,
    #     print_log=True,
    #     action_space_type='continuous',
    # )

    env = iGibsonEnv(configs=config_file, physics_timestep=1 / 120., action_timestep=1 / 30.)
    eval_env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=tensorboard_log_dir,
    #     policy_kwargs=policy_kwargs,
    #     n_steps=20*10,
    # )
    load_path = 'log_dir/20220504-113956/_51000_steps.zip'
    # model.load(load_path)
    # model.set_parameters(load_path)
    model = PPO.load(load_path)
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(name, param)
    # model.env = env
    print('Successfully loaded from {}'.format(load_path))
    log.debug(model.policy)
    print('Evaluating Started ...')
    # model.learn(1000000)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print('Evaluating Finished ...')
    log.info(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()