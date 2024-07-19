import omnigibson as og
from omnigibson.macros import gm
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.objects.object_base import BaseObject
from omnigibson.utils.config_utils import NumpyEncoder
from omnigibson.utils.python_utils import create_object_from_init_info
from omnigibson.utils.ui_utils import create_module_logger
import json
import h5py
from collections import defaultdict
import numpy as np
from pathlib import Path
import os
from copy import deepcopy

# Create module logger
log = create_module_logger(module_name=__name__)

h5py.get_config().track_order = True


class DataWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for writing data to an HDF5 file.
    """

    def __init__(self, env, output_path, only_successes=True):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file
            only_successes (bool): Whether to only save successful episodes
        """
        # Make sure the wrapped environment inherits correct omnigibson format
        assert isinstance(env, og.Environment), \
            "Expected wrapped @env to be a subclass of OmniGibson's Environment class!"

        # Only one scene is supported for now
        assert len(og.sim.scenes) == 1, "Only one scene is currently supported for DataWrapper env!"

        self.traj_count = 0
        self.step_count = 0
        self.max_state_size = 0
        self.only_successes = only_successes
        self.current_obs = None

        self.current_traj_history = []

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        log.info(f"\nWriting OmniGibson dataset hdf5 to: {output_path}\n")
        self.hdf5_file = h5py.File(output_path, 'w')
        data_grp = self.hdf5_file.create_group("data")
        env.task.write_task_metadata()
        scene_file = og.sim.save()[0]
        config = deepcopy(env.config)
        self.add_metadata(group=data_grp, name="config", data=config)
        self.add_metadata(group=data_grp, name="scene_file", data=scene_file)

        # Run super
        super().__init__(env=env)

    def step(self, action):
        """
        Run the environment step() function and collect data

        Args:
            action (np.array): action to take in environment

        Returns:
            5-tuple:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Aggregate step data
        step_data = self._parse_step_data(action, next_obs, reward, terminated, truncated, info)

        # Update obs and traj history
        self.current_traj_history.append(step_data)
        self.current_obs = next_obs

        return next_obs, reward, terminated, truncated, info

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        """
        Parse the output from the internal self.env.step() call and write relevant data to record to a dictionary

        Args:
            action (np.array): action deployed resulting in @obs
            obs (dict): state, i.e. observation
            reward (float): reward, i.e. reward at this current timestep
            terminated (bool): terminated, i.e. whether this episode ended due to a failure or success
            truncated (bool): truncated, i.e. whether this episode ended due to a time limit etc.
            info (dict): info, i.e. dictionary with any useful information

        Returns:
            dict: Keyword-mapped data that should be recorded in the HDF5
        """
        raise NotImplementedError()

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            2-tuple:
                - dict: Environment observation space after reset occurs
                - dict: Information related to observation metadata
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        self.current_obs, info = self.env.reset()
        return self.current_obs, info

    def observation_spec(self):
        """
        Grab the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()

    def process_traj_to_hdf5(self, traj_data, traj_grp_name):
        data_grp = self.hdf5_file.require_group("data")
        traj_grp = data_grp.create_group(traj_grp_name)
        traj_grp.attrs["num_samples"] = len(traj_data)

        data = {key: defaultdict(list) if "obs" in key else [] for key in traj_data[0]}

        for step_data in traj_data:
            for k, v in step_data.items():
                if "obs" in k:
                    for mod, step_mod_data in v.items():
                        data[k][mod].append(step_mod_data)
                else:
                    data[k].append(v)

        for k, dat in data.items():
            if "obs" in k:
                obs_grp = traj_grp.create_group(k)
                for mod, traj_mod_data in dat.items():
                    obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
            else:
                try:
                    traj_grp.create_dataset(k, data=np.stack(dat, axis=0))
                except ValueError:
                    breakpoint()

        return traj_grp

    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        # Only save successful demos and if actually recording
        success = self.env.task.success or not self.only_successes
        if success and self.hdf5_file is not None:
            traj_grp_name = f"demo_{self.traj_count}"
            traj_grp = self.process_traj_to_hdf5(self.current_traj_history, traj_grp_name)
            self.traj_count += 1
        else:
            # Remove this demo
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory and transition buffers
        self.current_traj_history = []

    def flush_current_file(self):
        self.hdf5_file.flush()  # Flush data to disk to avoid large memory footprint
        # Retrieve the file descriptor and use os.fsync() to flush to disk
        fd = self.hdf5_file.id.get_vfd_handle()
        os.fsync(fd)
        log.info("Flushing hdf5")

    def add_metadata(self, group, name, data):
        """
        Adds metadata to the current HDF5 file under the "data" key

        Args:
            group (hdf5.File or hdf5.Group): HDF5 object to add an attribute to
            name (str): Name to assign to the data
            data (str or dict): Data to add. Note that this only supports relatively primitive data types --
                if the data is a dictionary it will be converted into a string-json format using NumpyEncoder
        """
        group.attrs[name] = json.dumps(data, cls=NumpyEncoder) if isinstance(data, dict) else data

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        if self.hdf5_file is not None:

            log.info(f"\nSaved:\n"
                  f"{self.traj_count} trajectories / {self.step_count} total steps\n"
                  f"to hdf5: {self.hdf5_file.filename}\n")

            self.hdf5_file["data"].attrs["n_episodes"] = self.traj_count
            self.hdf5_file["data"].attrs["n_steps"] = self.step_count
            self.hdf5_file.close()


class DataCollectionWrapper(DataWrapper):
    """
    An OmniGibson environment wrapper for collecting data in an optimized way.

    NOTE: This does NOT aggregate observations. Please use DataPlaybackWrapper to aggregate an observation
    dataset!
    """

    def __init__(self, env, output_path, only_successes=True):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): path to store hdf5 data file
            only_successes (bool): Whether to only save successful episodes
        """
        # Store additional variables needed for optimized data collection
        self.max_state_size = 0
        self.current_transitions = dict()

        # Add callbacks on import / remove objects and systems
        og.sim.add_callback_on_system_init(name="data_collection", callback=lambda system: self.add_transition_info(obj=system, add=True))
        og.sim.add_callback_on_system_clear(name="data_collection", callback=lambda system: self.add_transition_info(obj=system, add=False))
        og.sim.add_callback_on_import_obj(name="data_collection", callback=lambda obj: self.add_transition_info(obj=obj, add=True))
        og.sim.add_callback_on_remove_obj(name="data_collection", callback=lambda obj: self.add_transition_info(obj=obj, add=False))

        # Run super
        super().__init__(env=env, output_path=output_path, only_successes=only_successes)

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store dumped state, reward, terminated, truncated
        step_data = dict()
        state = og.sim.dump_state(serialized=True)
        step_data["action"] = action
        step_data["state"] = state
        step_data["state_size"] = len(state)
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated

        # Update max state size
        self.max_state_size = max(self.max_state_size, len(state))

        return step_data

    def process_traj_to_hdf5(self, traj_data, traj_grp_name):
        # First pad all state values to be the same max (uniform) size
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = np.zeros(self.max_state_size, dtype=np.float32)
            padded_state[:len(state)] = state
            step_data["state"] = padded_state

        # Call super
        traj_grp = super().process_traj_to_hdf5(traj_data, traj_grp_name)

        # Add in transition info
        self.add_metadata(group=traj_grp, name="transitions", data=self.current_transitions)

        return traj_grp

    def flush_current_traj(self):
        # Call super first
        super().flush_current_traj()

        # Clear transition buffer and max state size
        self.max_state_size = 0
        self.current_transitions = dict()

    def add_transition_info(self, obj, add=True):
        """
        Adds transition info to the current sim step for specific object @obj.

        Args:
            obj (BaseObject or BaseSystem): Object / system whose information should be stored
            add (bool): If True, assumes the object is being imported. Else, assumes the object is being removed
        """
        if self.env.episode_steps not in self.current_transitions:
            self.current_transitions[self.env.episode_steps] = {"systems": {"add": [], "remove": []}, "objects": {"add": [], "remove": []}}

        # Add info based on type -- only need to store name unless we're an object being added
        info = obj.get_init_info() if isinstance(obj, BaseObject) and add else obj.name
        dic_key = "objects" if isinstance(obj, BaseObject) else "systems"
        val_key = "add" if add else "remove"
        self.current_transitions[self.env.episode_steps][dic_key][val_key].append(info)


class DataPlaybackWrapper(DataWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations.

    NOTE: This assumes a DataCollectionWrapper environment has been used to collect data!

    # TODO: Add params for setting camera width / height
    """

    @classmethod
    def create_from_hdf5(
        cls,
        input_path,
        output_path,
        robot_obs_modalities,
        external_obs_modalities,
        n_render_iterations=5,
        only_successes=False,
    ):
        """
        Create a DataPlaybackWrapper environment instance form the recorded demonstration info
        from @hdf5_path, and aggregate observation_modalities @obs during playback

        Args:
            input_path (str): Absolute path to the input hdf5 file containing the relevant collected data to playback
            output_path (str): Absolute path to the output hdf5 file that will contain the recorded observations from
                the replayed data
            robot_obs_modalities (list): Robot observation modalities to use. This list is directly passed into
                the robot_cfg (`obs_modalities` kwarg) when spawning the robot
            external_obs_modalities (list): External sensor observation modalities to use. This list is directly passed
                into all external sensors' `modalities` kwarg when spawning the external sensors
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            only_successes (bool): Whether to only save successful episodes

        Returns:
            DataPlaybackWrapper: Generated playback environment
        """
        # Read from the HDF5 file
        f = h5py.File(input_path, 'r')
        config = json.loads(f["data"].attrs["config"])

        # Hot swap in additional info for playing back data

        # Minimize physics leakage during playback (we need to take an env step when loading state)
        config["env"]["action_frequency"] = 1000.0
        config["env"]["physics_frequency"] = 1000.0

        # Make sure obs space is flattened for recording
        config["env"]["flatten_obs_space"] = True

        # Set scene file and disable online object sampling if BehaviorTask is being used
        config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False

        # for i, robot_cfg in enumerate(config["robots"]):
        #     robot_cfg["name"] = f"robot_new{i}"

        # Set observation modalities
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = robot_obs_modalities
        for external_sensor_cfg in config["env"]["external_sensors"]:
            external_sensor_cfg["modalities"] = external_obs_modalities

        # Load env
        env = og.Environment(configs=config)

        # Wrap and return env
        return cls(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            only_successes=only_successes,
        )

    def __init__(self, env, input_path, output_path, n_render_iterations=5, only_successes=False):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to store output hdf5 data file
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            only_successes (bool): Whether to only save successful episodes
        """
        # Make sure transition rules are DISABLED for playback since we manually propagate transitions
        assert not gm.ENABLE_TRANSITION_RULES, "Transition rules must be disabled for DataPlaybackWrapper env!"

        # Store scene file so we can restore the data upon each episode reset
        self.input_hdf5 = h5py.File(input_path, 'r')
        self.scene_file = json.loads(self.input_hdf5["data"].attrs["scene_file"])

        # Store additional variables
        self.n_render_iterations = n_render_iterations

        # Run super
        super().__init__(env=env, output_path=output_path, only_successes=only_successes)

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self.current_obs
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        return step_data

    def playback_episode(self, episode_id, record=True):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record (bool): Whether to record data during playback or not
        """
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        transitions = json.loads(traj_grp.attrs["transitions"])
        action = traj_grp["action"]
        state = traj_grp["state"]
        state_size = traj_grp["state_size"]
        reward = traj_grp["reward"]
        terminated = traj_grp["terminated"]
        truncated = traj_grp["truncated"]

        # Reset environment
        og.sim.restore(scene_files=[self.scene_file])
        self.reset()

        for i, (a, s, ss, r, te, tr) in enumerate(zip(
            action, state, state_size, reward, terminated, truncated
        )):
            # Execute any transitions that should occur at this current step
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.remove_system(remove_sys_name)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(np.ones(3) * 100.0 + np.ones(3) * 5 * j)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    og.sim.remove_object(obj)
                # Step physics to initialize any new objects
                og.sim.step()

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            og.sim.load_state(s[:int(ss)], serialized=True)
            self.current_obs, _, _, _, _ = self.env.step(action=a, n_render_iterations=self.n_render_iterations)

            # If recording, record data
            if record:
                step_data = dict()
                step_data["obs"] = self.current_obs
                step_data["action"] = a
                step_data["reward"] = r
                step_data["terminated"] = te
                step_data["truncated"] = tr
                self.current_traj_history.append(step_data)

            self.step_count += 1

        if record:
            self.flush_current_traj()

    def playback_dataset(self, record=True):
        """
        Playback all episodes from the input HDF5 file, and optionally record observation data if @record is True

        Args:
            record (bool): Whether to record data during playback or not
        """
        for episode_id in range(self.input_hdf5["data"].attrs["n_episodes"]):
            self.playback_episode(episode_id=episode_id, record=record)
