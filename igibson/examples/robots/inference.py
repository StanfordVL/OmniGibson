# https://github.com/StanfordVL/behavior/blob/main/behavior/baselines/rl/stable_baselines3_ppo_training.py

import logging
import os, cv2
from typing import Callable
import numpy as np
import torch
import random
import pdb

log = logging.getLogger(__name__)

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    # from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        self.debug_length = 10
        self.debug_mode = True
        self.random_cropping = True
        self.random_cropping_size = 116
        extractors = {}
        self.step_index = 0
        self.img_save_dir = 'img_save_dir'
        os.makedirs(self.img_save_dir, exist_ok=True)
        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():

            if key in ["rgb", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 8, kernel_size=8, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], self.random_cropping_size, self.random_cropping_size])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["accum_reward", 'obj_joint', 'obj_in_hand']:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size))
            else:
                continue
            total_concat_size += feature_size
        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1

        for key, extractor in self.extractors.items():
            if key in ["rgb",]:
                if self.random_cropping:
                    random_x = random.randint(0, 12)
                    random_y = random.randint(0, 12)
                    observations[key] = observations[key][:, random_x:(random_x+self.random_cropping_size), random_y:(random_y+self.random_cropping_size)]
                    observations[key] += torch.randn(observations[key].size()).cuda() * 0.05 + 0.0

                    observations[key] = torch.clamp(observations[key], 0.0, 1.0)
                if self.debug_mode:
                    cv2.imwrite(os.path.join(self.img_save_dir, '{0:06d}.png'.format(self.step_index % self.debug_length)), cv2.cvtColor((observations[key][0].cpu().numpy()*255).astype('uint8'), cv2.COLOR_RGB2BGR))
                observations[key] = observations[key].permute((0, 3, 1, 2))  # range: [0, 1]
            elif key in ["ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2)) / 500. # range: [0, 1]
            elif key in ['accum_reward', 'obj_joint', 'obj_in_hand']:
                if len(observations[key].shape) == 3:
                    observations[key] = observations[key].squeeze(-1)  # [:, :, 0]
            else:
                continue

            encoded_tensor_list.append(extractor(observations[key]))
        feature = th.cat(encoded_tensor_list, dim=1)

        return feature

def evaluate_policy(
    model,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    return_episode_rewards: bool = False,
    observation_space = (128, 128, 3),
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        obs = {'rgb': np.ones((1,) + observation_space),
               'obj_in_hand': np.ones((1, 1))}
        # Because recurrent policies need the same observation space during training and evaluation, we need to pad
        # observation to match training shape. See https://github.com/hill-a/stable-baselines/issues/1015
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            # Execute action here and return info (possibly reward, done)
            # Here are placeholders for obs, reward, done, _info
            # new_obs, reward, done, _info = env.step(action)
            reward = 1.
            if episode_length > 10:
                done = True
            else:
                done = False
            _info = {}
            # Obtain new observation here:
            new_obs = {'rgb': np.ones((1,) + observation_space),
                       'obj_in_hand': np.ones((1, 1))}
            obs = new_obs
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def main():
    seed = 0
    # Load model, remember to use the correct network architecture
    load_path = 'log_dir/20220609-072359/_15000_steps'
    # pdb.set_trace()
    model = PPO.load(load_path)
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(name, param)
    print('Successfully loaded from {}'.format(load_path))
    log.debug(model.policy)
    print('Evaluating Started ...')
    mean_reward, std_reward = evaluate_policy(model, n_eval_episodes=50)
    print('Evaluating Finished ...')
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()