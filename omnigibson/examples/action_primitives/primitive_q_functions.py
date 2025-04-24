import argparse
import time

import numpy as np
# import torch
from torch import nn
from torch.utils.data import Dataset

import omnigibson as og
from omnigibson.envs.cut_pour_pkg_in_bowl import CutPourPkgInBowlEnv
from omnigibson.envs.pack_gift import PackGiftEnv
from omnigibson.envs.primitives_env import PrimitivesEnv

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
    task_env_kwargs = dict(
        out_dir="/home/albert/dev/OmniGibson/out_videos",
        obj_to_grasp_name="box",
        vid_speedup=args.vid_speedup,
    )
    if args.env == "CutPourPkgInBowl":
        task_env = CutPourPkgInBowlEnv(**task_env_kwargs)
    elif args.env == "PackGift":
        task_env = PackGiftEnv(**task_env_kwargs)
    else:
        raise NotImplementedError
    env = PrimitivesEnv(
        task_env, max_path_len=max_path_len, debug=args.debug)
    data_collector = ScriptedDataCollector(env, "CutPourPkgInBowlEnv", max_path_len=max_path_len, args=args)
    print(f"time to init env: {time.time() - st}")

    st = time.time()
    for i in range(10):
        og.sim.step()
    print(f"time for 100 sim steps: {time.time() - st}")
    st = time.time()
    env.make_video()
    obs, info = env.reset()
    print(f"time for env reset: {time.time() - st}")

    # Randomize the robot pose
    # robot.states[OnTop].set_value(floor, True)

    # Randomize the apple pose on top of the breakfast table
    # apple.states[OnTop].set_value(breakfast_table, True)

    data_collector.collect_trajs(n=args.n)
    for primitive_int, total_num_env_steps_list in (
            data_collector.primitive_int_to_num_env_steps_list_map.items()):
        avg_num_ts = np.nanmean(np.array(total_num_env_steps_list, dtype=np.float64))
        print(
            f"total_num_env_steps_list for primitive {primitive_int}",
            total_num_env_steps_list,
            f"Avg: {avg_num_ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=str, default="/home/albert/dev/OmniGibson/out_hdf5")
    parser.add_argument(
        "--n", type=int, default=1)
    parser.add_argument(
        "--debug", default=False, action="store_true")
    parser.add_argument(
        "--env", choices=["CutPourPkgInBowl", "PackGift"], required=True)
    parser.add_argument(
        "--vid-speedup", type=int, default=2)
    parser.add_argument(
        "--task-ids", type=str, default="")  # comma separated list of task ids
    args = parser.parse_args()
    args.task_ids = eval(f"[{args.task_ids}]")
    args.no_vid = bool(args.vid_speedup == 0)
    main(args)
