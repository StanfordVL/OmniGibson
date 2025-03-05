import argparse
import time

import numpy as np
# import torch
from torch import nn
from torch.utils.data import Dataset

import omnigibson as og
from omnigibson.envs.primitives_env import (
    CutPourPkgInBowlEnv,
    PrimitivesEnv,
)

from omnigibson.utils.data_collection_utils import ScriptedDataCollector


class QFnDataset(Dataset):
    pass
    # TODO


class QFnRobotEstimator(nn.Module):
    pass
    # TODO


class QFnEstimatorTrainer:
    pass
    # TODO


def main(args):
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.

    It loads Rs_int with a robot, and the robot picks and places an apple.
    """
    # Load the config

    st = time.time()
    max_path_len = 1
    task_env = CutPourPkgInBowlEnv(out_dir="/home/albert/dev/OmniGibson/out_videos", obj_to_grasp_name="box")
    env = PrimitivesEnv(task_env, max_path_len=max_path_len)
    data_collector = ScriptedDataCollector(env, "CutPourPkgInBowlEnv", max_path_len=max_path_len, args=args)
    print(f"time to init env: {time.time() - st}")

    st = time.time()
    for i in range(10):
        og.sim.step()
    print(f"time for 100 sim steps: {time.time() - st}")
    st = time.time()
    env.make_video()
    env.scene.reset()
    obs, info = env.reset()
    print(f"time for env reset: {time.time() - st}")

    # Randomize the rbot pose
    # robot.states[OnTop].set_value(floor, True)

    # Randomize the apple pose on top of the breakfast table
    # apple.states[OnTop].set_value(breakfast_table, True)

    data_collector.collect_trajs(n=2)
    total_num_env_steps_list = data_collector.primitive_int_to_num_env_steps_list_map[0]

    avg_num_ts = np.nanmean(np.array(total_num_env_steps_list, dtype=np.float64))
    print("total_num_env_steps_list", total_num_env_steps_list, f"Avg: {avg_num_ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=str, default="/home/albert/dev/OmniGibson/out_hdf5")
    parser.add_argument(
        "--n", type=int, default=1)
    args = parser.parse_args()
    main(args)
