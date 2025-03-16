import collections
import json
import os
import time

import datetime
import h5py
import numpy as np
from tqdm import tqdm


class ScriptedDataCollector:
    def __init__(
            self,
            # Either pass in these args, for use during training
            env=None,
            env_name="",
            max_path_len=None,
            # OR pass in these two args, for use as standalone script.
            args={},
    ):
        if max_path_len:
            self.max_path_len = max_path_len
        else:
            self.max_path_len = args.max_steps

        self.args = args
        self.env = env
        self.env_name = env_name
        self.out_dir = os.path.join(args.out_dir, get_timestamp_str())
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.ds_keys = [
            "observations", "actions", "rewards", "next_observations",
            "terminals", "infos", "next_infos", "num_steps_info"]
        self.obs_keys = ["state", "symb_state"]
        self.next_obs_keys = ["state", "symb_state"]
        self.task_ids = env.task_ids

        self.primitive_int_to_rew_list_map = {}
        self.primitive_int_to_num_env_steps_list_map = {}

    def init_empty_traj_dict(self):
        traj_dict = {}
        for ds_key in self.ds_keys:
            if ds_key in ["observations", "next_observations"]:
                traj_dict[ds_key] = {}
            else:
                traj_dict[ds_key] = []

        for obs_key in self.obs_keys:
            traj_dict["observations"][obs_key] = []

        for next_obs_key in self.next_obs_keys:
            traj_dict["next_observations"][next_obs_key] = []

        return traj_dict

    def add_transition(
            self,
            traj_dict,
            obs_dict,
            action,
            r,
            next_obs_dict,
            done,
            info,
            next_info,
            num_steps_info,
    ):
        for obs_key in self.obs_keys:
            if isinstance(obs_dict[obs_key], list):
                traj_dict["observations"][obs_key].extend(obs_dict[obs_key])
            else:
                traj_dict["observations"][obs_key].append(obs_dict[obs_key])
        for next_obs_key in self.next_obs_keys:
            if isinstance(next_obs_dict[next_obs_key], list):
                traj_dict["next_observations"][next_obs_key].extend(
                    next_obs_dict[next_obs_key])
            else:
                traj_dict["next_observations"][next_obs_key].append(
                    next_obs_dict[next_obs_key])

        traj_dict["actions"].append(action)
        traj_dict["rewards"].append(r)
        traj_dict["terminals"].append(done)
        traj_dict["infos"].append(info)
        traj_dict["next_infos"].append(next_info)
        traj_dict["num_steps_info"].append(num_steps_info)

    def collect_single_traj(self, task_id=0):
        # may be an unsuccessful traj
        # Init traj datastructures
        MAX_TS = 10000
        traj_dict = self.init_empty_traj_dict()

        obs, info = self.env.reset(return_info=True)

        for t in range(self.max_path_len):
            stf = time.time()
            action = [1][t]  # Put the plan here. TODO: make sure env action space maps to primitives.

            self.env.pre_step_obj_loading(action)
            if self.args.no_vid:
                next_obs, r, done, next_info = self.env.step(action)
            else:
                try:
                    next_obs, r, done, next_info = self.env.step(action)
                except Exception as e:
                    print(f"env step failed on action {action}")
                    print(f"Got error:\n{e}")
                    skill_info = {}
                    skill_info['skill_success'] = False
                    skill_info['num_steps_info'] = dict(
                        total_env_steps=MAX_TS,
                        total_env_steps_wo_teleport=MAX_TS,
                    )

                    skill_name, _ = self.env.skill_name_param_action_space[action]
                    if skill_name == "pickplace":  # TODO fix this to get the primitive name
                        skill_info['num_steps_info'].update(dict(
                            nav_to_place_teleport_aerial_dist=np.nan,
                            nav_to_place_teleport_speed=0.01,
                            nav_to_place_teleport_inferred_steps=np.nan,
                        ))
                    next_obs, r, done, next_info = self.env.post_step(skill_info)
                    next_info['skill_success'] = bool(r)
            print(f"Total time to execute trajectory: {time.time() - stf}")

            self.add_transition(
                traj_dict, obs, action, r, next_obs, done, info, next_info,
                next_info['num_steps_info'])
            obs = next_obs
            info = next_info

            # store skill reward and timesteps.
            skill_success = bool(info['skill_success'])
            assert skill_success == r
            num_env_steps = info['num_steps_info']['total_env_steps']

            if action not in self.primitive_int_to_rew_list_map:
                self.primitive_int_to_rew_list_map[action] = []
            self.primitive_int_to_rew_list_map[action].append(
                skill_success)

            if action not in self.primitive_int_to_num_env_steps_list_map:
                self.primitive_int_to_num_env_steps_list_map[action] = []
            self.primitive_int_to_num_env_steps_list_map[action].append(
                num_env_steps)

            self.print_success_rate_by_skill()

        self.env.make_video()
        self.save_single_demo_hdf5(traj_dict, task_id)

    def save_single_demo_hdf5(self, traj_dict, task_id):
        # Create single-demo hdf5 file.
        demo_file_name = f"{self.out_dir}/{get_timestamp_str()}.hdf5"
        print(demo_file_name)
        demo_file = h5py.File(demo_file_name, "w")

        grp = demo_file.create_group("data")
        self.task_to_grp_map = dict(
            [(task_id, grp.create_group(str(task_id)))
                for task_id in self.task_ids])

        # Save the dataset group
        task_id_grp = self.task_to_grp_map[task_id]

        traj_idx = self.task_id_to_num_trajs_map[task_id]
        ep_grp = task_id_grp.create_group(f"demo_{traj_idx}")
        self.task_id_to_num_trajs_map[task_id] += 1

        obs_grp = ep_grp.create_group("observations")
        next_obs_grp = ep_grp.create_group("next_observations")

        for obs_key in self.obs_keys:
            obs_grp.create_dataset(
                obs_key, data=np.stack(traj_dict["observations"][obs_key]))
        for next_obs_key in self.next_obs_keys:
            next_obs_grp.create_dataset(
                next_obs_key,
                data=np.stack(traj_dict["next_observations"][next_obs_key]))

        ep_grp.create_dataset("actions", data=np.stack(traj_dict["actions"]))
        ep_grp.create_dataset("rewards", data=np.stack(traj_dict["rewards"]))
        ep_grp.create_dataset(
            "terminals", data=np.stack(traj_dict["terminals"]))
        ep_grp.create_dataset(
            "num_env_steps", data=np.stack(
                [x["total_env_steps"] for x in traj_dict["num_steps_info"]]))
        ep_grp.create_dataset(
            "num_env_steps_wo_teleport", data=np.stack(
                [x.get("total_env_steps_wo_teleport", x["total_env_steps"])
                 for x in traj_dict["num_steps_info"]]))
        ep_grp.create_dataset(
            "teleport_aerial_dist", data=np.stack(
                [x.get("teleport_aerial_dist", 0)
                 for x in traj_dict["num_steps_info"]]))

        # save info, next_info, and num_steps_info
        infos_grp = ep_grp.create_group("infos")
        infos_grp.attrs["dicts_by_ts"] = str(traj_dict["infos"]).replace(
            "array(", "np.array(")
        next_infos_grp = ep_grp.create_group("next_infos")
        next_infos_grp.attrs["dicts_by_ts"] = str(traj_dict["next_infos"]).replace(
            "array(", "np.array(")
        num_steps_infos_grp = ep_grp.create_group("num_steps_info")
        num_steps_infos_grp.attrs["dicts_by_ts"] = str(traj_dict["num_steps_info"]).replace(
            "array(", "np.array(")

        self.demo_fpaths.append(demo_file_name)

    def collect_trajs(self, n):
        self.task_id_to_num_trajs_map = collections.Counter()
        os.makedirs(f"{self.out_dir}", exist_ok=True)
        self.demo_fpaths = []

        for i in tqdm(range(n)):
            self.collect_single_traj(task_id=self.task_ids[0])

        self.env.reset()

        # Save to a single demo file.
        # Concat everything in self.demo_fpaths
        env_info = {}
        print("self.demo_fpaths", self.demo_fpaths)
        out_path = concat_hdf5(
            self.demo_fpaths, self.out_dir, env_info, self.env_name)

        print("TaskID --> # trajs collected", self.task_id_to_num_trajs_map)

    def print_success_rate_by_skill(self):
        num_trajs = 0
        out_str = "Number of successes so far (by action):\n"
        for action, rew_list in self.primitive_int_to_rew_list_map.items():
            num_trajs += len(rew_list)
            if len(rew_list):
                out_str += f"- {action}: {np.sum(rew_list)}/{len(rew_list)}"
        if num_trajs:
            print(out_str)
        return out_str


def get_timestamp_str(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider))


def concat_hdf5(
        hdf5_list, out_dir, env_info, env_name,
        save_orig_hdf5_list=False, demo_attrs_to_del=[]):
    timestamp = get_timestamp_str()
    out_path = os.path.join(out_dir, f"scripted_{env_name}_{timestamp}.hdf5")
    f_out = h5py.File(out_path, mode='w')
    grp = f_out.create_group("data")

    task_idx_to_num_eps_map = collections.Counter()
    env_args = None
    for h5name in tqdm(hdf5_list):
        print(h5name)
        h5fr = h5py.File(h5name, 'r')
        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for task_idx in h5fr['data'].keys():
            if task_idx not in f_out['data'].keys():
                task_idx_grp = grp.create_group(task_idx)
            else:
                task_idx_grp = f_out[f'data/{task_idx}']
            task_idx = int(task_idx)
            for demo_id in h5fr[f'data/{task_idx}'].keys():
                # Set lang_list under task_idx grp
                # if "lang_list" not in task_idx_grp.attrs.keys():
                #     if "lang_list" in h5fr[f'data/{task_idx}'].attrs.keys():
                #         task_idx_grp.attrs["lang_list"] = (
                #             h5fr[f'data/{task_idx}'].attrs["lang_list"])
                #     else:
                #         task_idx_grp.attrs["lang_list"] = [
                #             str(x) for x in (
                #                 h5fr[f'data/{task_idx}/{demo_id}']
                #                 .attrs['lang_list'])]
                # if "is_multistep" not in task_idx_grp.attrs.keys():
                #     task_idx_grp.attrs["is_multistep"] = h5fr[
                #         f'data/{task_idx}'].attrs["is_multistep"]

                task_idx_traj_num = task_idx_to_num_eps_map[task_idx]
                h5fr.copy(
                    f"data/{task_idx}/{demo_id}", task_idx_grp,
                    name=f"demo_{task_idx_traj_num}")

                for attr_to_del in demo_attrs_to_del:
                    if attr_to_del in (
                            task_idx_grp[f'demo_{task_idx_traj_num}']
                            .attrs.keys()):
                        del (task_idx_grp[f"demo_{task_idx_traj_num}"]
                             .attrs[attr_to_del])
                task_idx_to_num_eps_map[task_idx] += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    if env_args is not None:
        grp.attrs["env_args"] = env_args
    grp.attrs["env_info"] = str(env_info)
    print("env_info", env_info)
    if save_orig_hdf5_list:
        hdf5_list_dict = json.dumps(dict(
            zip(range(len(hdf5_list)), sorted(hdf5_list))))
        grp.attrs["orig_hdf5_list"] = hdf5_list_dict

    print("saved to", out_path)
    f_out.close()

    return out_path
