# https://github.com/StanfordVL/behavior/blob/main/behavior/baselines/rl/stable_baselines3_ppo_training.py

import logging
import os, time
from typing import Callable
import pdb

import igibson
# import behavior
# from igibson.envs.skill_env import SkillEnv
from igibson.envs.igibson_env import iGibsonEnv
# from igibson.sensors.vision_sensor import VisionSensor
from igibson.wrappers import ActionPrimitiveWrapper
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
    # from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

except ModuleNotFoundError:
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

#
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            # if key in ["proprioception", "task_obs"]:
            #     extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            # elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
            #     n_input_channels = subspace.shape[2]  # channel last
            #     cnn = nn.Sequential(
            #         nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            #         nn.ReLU(),
            #         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            #         nn.ReLU(),
            #         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            #         nn.ReLU(),
            #         nn.Flatten(),
            #     )
            #     test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
            #     with th.no_grad():
            #         n_flatten = cnn(test_tensor[None]).shape[1]
            #     fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            #     extractors[key] = nn.Sequential(cnn, fc)
            # elif key in ["scan"]:
            #     n_input_channels = subspace.shape[1]  # channel last
            #     cnn = nn.Sequential(
            #         nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            #         nn.ReLU(),
            #         nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
            #         nn.ReLU(),
            #         nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
            #         nn.ReLU(),
            #         nn.Flatten(),
            #     )
            #     test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
            #     with th.no_grad():
            #         n_flatten = cnn(test_tensor[None]).shape[1]
            #     fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            #     extractors[key] = nn.Sequential(cnn, fc)
            # elif key in ['occupancy_grid']:
            #     continue
            # elif key == "state_vec":
            #     # Run through a simple MLP
            #     extractors[key] = nn.Linear(subspace.shape[0], 16)
            #     total_concat_size += 16
            # else:
            #     raise ValueError("Unknown observation key: %s" % key)

            if key in ["rgb", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["accum_reward", 'obj_joint']:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size))
            else:
                continue
            total_concat_size += feature_size
        # if 'accum_reward' in observation_space.spaces:
        #     extractors['accum_reward'] = nn.Sequential()
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # if key in ['occupancy_grid']:
            #     continue
            # if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
            #     observations[key] = observations[key].permute((0, 3, 1, 2))
            # elif key in ["scan"]:
            #     observations[key] = observations[key].permute((0, 2, 1))
            # print(key, observations[key])  # [0, 500]
            if key in ["rgb",]:
                observations[key] = observations[key].permute((0, 3, 1, 2))  # range: [0, 1]
            elif key in ["ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2)) / 500. # range: [0, 1]
            elif key in ['accum_reward', 'obj_joint']:
                # print('observations[key].shape: ', observations[key].shape)
                # if len(observations[key]) == 1:
                #     observations[key] = observations[key][None, :]
                if len(observations[key].shape) == 3:
                    observations[key] = observations[key].squeeze(-1)  # [:, :, 0]
            else:
                continue
            # print(self.extractors)
            # print('observations[key].shape: ', observations[key].shape)

            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # if 'accum_reward' in self.extractors:
        #     print('observations[accum_reward].shape, encoded_tensor_list[0].shape: ', observations['accum_reward'].shape,
        #           observations['ins_seg'].shape, encoded_tensor_list[0].shape)
            # observations[accum_reward].shape, encoded_tensor_list[0].shape:  torch.Size([1]) torch.Size([1, 128])
            # encoded_tensor_list.append(observations['accum_reward'][])
        feature = th.cat(encoded_tensor_list, dim=1)
        # print('feature.shape: ', feature.shape)  #  torch.Size([1, 384])
        return feature


def main():
    # config_file = os.path.join('..', 'configs', "behavior_pick_and_place.yaml")
    # config_file = os.path.join('..', '..', 'configs', 'robots', "fetch_rl.yaml")
    # config_file = os.path.join(igibson.configs_path, "fetch_behavior_aps_putting_away_Halloween_decorations.yaml")
    config_file = f"{example_config_path}/behavior_mp_tiago.yaml"  # os.path.join(igibson.configs_path, "fetch_rl_cleaning_microwave_oven.yaml")
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ''
    num_cpu = 1
    mode = 'chen'  # 'callback'

    # def make_env(rank: int, seed: int = 0, print_log=True) -> Callable:
    #     def _init() -> ActionPrimitiveWrapper:
    #         env = iGibsonEnv(configs=config_file, physics_timestep=1 / 120., action_timestep=1 / 30.)
    #         env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives")
    #         env.seed(seed + rank)
    #         # env = lambda env=iGibsonEnv(configs=config_file, physics_timestep=1 / 120., action_timestep=1 / 30.): ActionPrimitiveWrapper(env=env)
    #         return env
    #
    #     set_random_seed(seed)
    #     return _init

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # env = VecMonitor(env)
    # env = SubprocVecEnv([make_env(0)])

    seed = 0

    env = iGibsonEnv(configs=config_file, physics_timestep=1 / 120., action_timestep=1 / 30.)
    env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives")
    ceiling = env.scene.object_registry("name", "ceilings")
    ceiling.visible = False
    env.seed(seed)
    set_random_seed(seed)
    env.reset()

    # eval_env = SkillEnv(
    #     config_file=config_file,
    #     mode="gui_interactive",
    #     action_timestep=1 / 30.0,
    #     physics_timestep=1 / 300.0,
    #     print_log=False,
    # )

    # eval_env = SubprocVecEnv([make_env(i, print_log=True) for i in range(num_cpu)])
    # eval_env = VecMonitor(eval_env)
    # eval_env = env

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)
    # pdb.set_trace()
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=20 * 10,
        batch_size=8,
        device='cuda',
    )
    # pdb.set_trace()
    # print('\n\n\n\n\n\n\n\n\n\n\n\n')
    log.debug(model.policy)

    if mode == 'for_loop':
        for i in range(1000):
            model.learn(1000)

            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
            log.info(f"After Training {i}: Mean reward: {mean_reward} +/- {std_reward:.2f}")

            model.save(os.path.join(tensorboard_log_dir, "ckpt_{:03d}_{}".format(i, prefix)))
        del model
    elif mode == 'callback':
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir,
                                             name_prefix=prefix)
        eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(tensorboard_log_dir, 'best_model'),
                                     log_path=os.path.join(tensorboard_log_dir, 'results'), eval_freq=1000)
        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])

        model.learn(1000000, callback=callback)

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        log.info(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

        model.save(os.path.join(tensorboard_log_dir, "ckpt_{}".format(prefix)))
        del model
    else:
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        log.debug(model.policy)
        print(model)

        model.learn(total_timesteps=10000000, callback=checkpoint_callback,
                    eval_env=env, eval_freq=1000,
                    n_eval_episodes=20)

    model = PPO.load(os.path.join(tensorboard_log_dir, "ckpt_{}".format(prefix)))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    log.info(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()