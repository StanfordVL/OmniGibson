import gym
import logging
from collections import OrderedDict, Iterable

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.sim_utils import get_collisions
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.python_utils import assert_valid_key, merge_nested_dicts, create_class_from_registry_and_config, \
    Serializable, Recreatable

import numpy as np
from transforms3d.euler import euler2quat
from igibson.robots.robot_base import BaseRobot

# Create settings for this module
m = create_module_macros(module_path=__file__)

# How many predefined randomized scene object configurations we have per scene
m.N_PREDEFINED_OBJ_RANDOMIZATIONS = 10


class Environment(gym.Env, GymObservable, Recreatable):
    """
    Core environment class that handles loading scene, robot(s), and task, following OpenAI Gym interface.
    """
    def __init__(
        self,
        configs,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device=None,
        automatic_reset=False,
    ):
        """
        :param configs (str or dict or list of str or dict): config_file path(s) or raw config dictionaries.
            If multiple configs are specified, they will be merged sequentially in the order specified.
            This allows procedural generation of a "full" config from small sub-configs.
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device: None or str, specifies the device to be used if running on the gpu with torch backend
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        # Call super first
        super().__init__()

        # Store settings and other initialized values
        self._automatic_reset = automatic_reset
        self._predefined_object_randomization_idx = 0
        self._n_predefined_object_randomizations = m.N_PREDEFINED_OBJ_RANDOMIZATIONS
        self.action_timestep = action_timestep

        # Initialize other placeholders that will be filled in later
        self._ignore_robot_object_collisions = None         # This will be a list of len(scene.robots) containing sets of objects to ignore collisions for
        self._ignore_robot_self_collisions = None           # This will be a list of len(scene.robots) containing sets of RigidPrims (robot's links) to ignore collisions for
        self._initial_pos_z_offset = None                   # how high to offset object placement to account for one action step of dropping
        self._texture_randomization_freq = None
        self._object_randomization_freq = None
        self._task = None
        self._scene = None
        self._loaded = None
        self._current_episode = 0

        # Variables reset at the beginning of each episode
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

        # Convert config file(s) into a single parsed dict
        configs = configs if isinstance(configs, list) or isinstance(configs, tuple) else [configs]

        # Initial default config
        self.config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=self.config, extra_dict=parse_config(config), inplace=True)
        # import pdb
        # pdb.set_trace()
        self.output = self.config["robots"][0]["obs_modalities"]
        # Set the simulator settings
        og.sim.set_simulation_dt(physics_dt=physics_timestep, rendering_dt=action_timestep)
        og.sim.viewer_width = self.render_config["viewer_width"]
        og.sim.viewer_height = self.render_config["viewer_height"]
        og.sim.vertical_fov = self.render_config["vertical_fov"]
        og.sim.device = device

        # Load this environment
        self.load()

    def add_ignore_robot_object_collision(self, robot_idn, obj):
        """
        Add a new robot-object pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            obj (BaseObject): Which object to ignore a collision for
        """
        self._ignore_robot_object_collisions[robot_idn].add(obj)

    def add_ignore_robot_self_collision(self, robot_idn, link):
        """
        Add a new robot-link self pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            link (RigidPrim): Which robot link to ignore a collision for
        """
        self._ignore_robot_self_collisions[robot_idn].add(link)

    def remove_ignore_robot_object_collision(self, robot_idn, obj):
        """
        Remove a robot-object pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            obj (BaseObject): Which object to no longer ignore a collision for
        """
        # self._ignore_robot_object_collisions[robot_idn].remove(obj)
        try:
            self._ignore_robot_object_collisions[robot_idn].remove(obj)
        except:
            return

    def remove_ignore_robot_self_collision(self, robot_idn, link):
        """
        Remove a robot-link self pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            link (RigidPrim): Which robot link to no longer ignore a collision for
        """
        self._ignore_robot_self_collisions[robot_idn].remove(link)

    def reload(self, configs, overwrite_old=True):
        """
        Reload using another set of config file(s).
        This allows one to change the configuration and hot-reload the environment on the fly.

        Args:
            configs (str or list of str): config_file path(s). If multiple configs are specified, they will
                be merged sequentially in the order specified. This allows procedural generation of a "full" config from
                small sub-configs.
            overwrite_old (bool): If True, will overwrite the internal self.config with @configs. Otherwise, will
                merge in the new config(s) into the pre-existing one. Setting this to False allows for minor
                modifications to be made without having to specify entire configs during each reload.
        """
        # Convert config file(s) into a single parsed dict
        configs = [configs] if isinstance(configs, str) else configs

        # Initial default config
        new_config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=new_config, extra_dict=parse_config(config), inplace=True)

        # Either merge in or overwrite the old config
        if overwrite_old:
            self.config = new_config
        else:
            merge_nested_dicts(base_dict=self.config, extra_dict=new_config, inplace=True)

        # Load this environment again
        self.load()

    def reload_model(self, scene_model):
        """
        Reload another scene model.
        This allows one to change the scene on the fly.

        :param scene_model: new scene model to load (eg.: Rs_int)
        """
        self.scene_config["model"] = scene_model
        self.load()

    def _load_variables(self):
        """
        Load variables from config
        """
        # Store additional variables after config has been loaded fully
        self._initial_pos_z_offset = self.env_config["initial_pos_z_offset"]
        self._texture_randomization_freq = self.env_config["texture_randomization_freq"]
        self._object_randomization_freq = self.env_config["object_randomization_freq"]

        # Set other values
        self._ignore_robot_object_collisions = [set() for _ in self.robots_config]
        self._ignore_robot_self_collisions = [set() for _ in self.robots_config]

        # Reset bookkeeping variables
        self._reset_variables()
        self._current_episode = 0           # Manually set this to 0 since resetting actually increments this

        # - Potentially overwrite the USD entry for the scene if none is specified and we're online sampling -

        # Make sure the requested scene is valid
        scene_type = self.scene_config["type"]
        assert_valid_key(key=scene_type, valid_keys=REGISTERED_SCENES, name="scene type")

        # If we're using a BehaviorTask, we may load a pre-cached scene configuration
        if self.task_config["type"] == "BehaviorTask":
            usd_file, usd_path = self.scene_config["usd_file"], self.scene_config["usd_path"]
            if usd_path is None and usd_file is None and not self.task_config["online_object_sampling"]:
                usd_file = "{}_task_{}_{}_{}_fixed_furniture_template".format(
                    self.scene_config["scene_model"],
                    self.task_config["activity_name"],
                    self.task_config["activity_definition_id"],
                    self.task_config["activity_instance_id"],
                )
            # Update the value in the scene config
            self.scene_config["usd_file"] = usd_file

        # - Additionally run some sanity checks on these values -

        # Check to make sure our z offset is valid -- check that the distance travelled over 1 action timestep is
        # less than the offset we set (dist = 0.5 * gravity * (t^2))
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self._initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

    def _load_task(self):
        """
        Load task
        """
        # Sanity check task to make sure it's valid
        task_type = self.task_config["type"]
        assert_valid_key(key=task_type, valid_keys=REGISTERED_TASKS, name="task type")

        # Grab the kwargs relevant for the specific task and create the task
        self._task = create_class_from_registry_and_config(
            cls_name=self.task_config["type"],
            cls_registry=REGISTERED_TASKS,
            cfg=self.task_config,
            cls_type_descriptor="task",
        )

        assert og.sim.is_stopped(), "sim should be stopped when load_task starts"
        og.sim.play()

        # Load task. Should load additinal task-relevant objects and configure the scene into its default initial state
        self._task.load(env=self)

        # Update the initial scene state
        self._scene.update_initial_state()

        og.sim.stop()

    def _load_scene(self):
        """
        Load the scene and robot specified in the config file.
        """
        # Create the scene from our scene config
        scene = create_class_from_registry_and_config(
            cls_name=self.scene_config["type"],
            cls_registry=REGISTERED_SCENES,
            cfg=self.scene_config,
            cls_type_descriptor="scene",
        )
        og.sim.import_scene(scene)

        # Save scene internally
        self._scene = scene

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        if len(self._scene.robots) == 0:
            # Iterate over all robots to generate in the robot config
            for i, robot_config in enumerate(self.robots_config):
                # Add a name for the robot if necessary
                if "name" not in robot_config:
                    robot_config["name"] = f"robot{i}"
                # Set prim path
                robot_config["prim_path"] = f"/World/{robot_config['name']}"
                # Make sure robot exists, grab its corresponding kwargs, and create / import the robot
                robot = create_class_from_registry_and_config(
                    cls_name=robot_config["type"],
                    cls_registry=REGISTERED_ROBOTS,
                    cfg=robot_config,
                    cls_type_descriptor="robot",
                )
                # Import the robot into the simulator
                og.sim.import_object(robot)

    def _load_observation_space(self):
        # Grab robot(s) and task obs spaces
        obs_space = OrderedDict()

        for robot in self.robots:
            # Load the observation space for the robot
            obs_space[robot.name] = robot.load_observation_space()

        # Also load the task obs space
        obs_space["task"] = self._task.load_observation_space()
        return obs_space

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.
        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    # def load_observation_space(self):
    #     self.image_width = self.config.get("image_width", 128)
    #     self.image_height = self.config.get("image_height", 128)
    #     observation_space = OrderedDict()
    #     sensors = OrderedDict()
    #     vision_modalities = []
    #     scan_modalities = []
    #     if "task_obs" in self.output:
    #         observation_space["task_obs"] = self.build_obs_space(
    #             shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
    #         )
    #     if "rgb" in self.output:
    #         observation_space["rgb"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
    #         )
    #         vision_modalities.append("rgb")
    #     if "depth" in self.output:
    #         observation_space["depth"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
    #         )
    #         vision_modalities.append("depth")
    #     if "pc" in self.output:
    #         observation_space["pc"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
    #         )
    #         vision_modalities.append("pc")
    #     if "optical_flow" in self.output:
    #         observation_space["optical_flow"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
    #         )
    #         vision_modalities.append("optical_flow")
    #     if "scene_flow" in self.output:
    #         observation_space["scene_flow"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
    #         )
    #         vision_modalities.append("scene_flow")
    #     if "normal" in self.output:
    #         observation_space["normal"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
    #         )
    #         vision_modalities.append("normal")
    #     if "seg" in self.output:
    #         observation_space["seg"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
    #         )
    #         vision_modalities.append("seg")
    #     if "ins_seg" in self.output:
    #         observation_space["ins_seg"] = self.build_obs_space(
    #             shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
    #         )
    #         vision_modalities.append("ins_seg")
    #
    #     self.observation_space = gym.spaces.Dict(observation_space)
    #     self.sensors = sensors
    #     return self.observation_space

    def _load_action_space(self):
        """
        Load action space for each robot
        """
        self.action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self.robots})

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        self._load_robots()
        self._load_task()

        og.sim.play()
        self.reset()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Denote that the scene is loaded
        self._loaded = True

    def reload_model_object_randomization(self, predefined_object_randomization_idx=None):
        """
        Reload the same model, with either @object_randomization_idx seed or the next object randomization random seed.

        Args:
            predefined_object_randomization_idx (None or int): If set, specifies the specific pre-defined object
                randomization instance to use. Otherwise, the current seed will be incremented by 1 and used.
        """
        assert self._object_randomization_freq is not None, \
            "object randomization must be active to reload environment with object randomization!"
        self._predefined_object_randomization_idx = predefined_object_randomization_idx if \
            predefined_object_randomization_idx is not None else \
            (self._predefined_object_randomization_idx + 1) % self._n_predefined_object_randomizations
        self.load()

    def close(self):
        """
        Clean up the environment and shut down the simulation.
        """
        og.sim.close()

    def get_obs(self):
        """
        Get the current environment observation.

        Returns:
            OrderedDict: Keyword-mapped observations, which are possibly nested
        """
        obs = OrderedDict()

        # Grab all observations from each robot
        for robot in self.robots:
            obs[robot.name] = robot.get_obs()

        # Add task observations
        obs["task"] = self._task.get_obs(env=self)

        return obs

    def _filter_collisions(self, collisions):
        """
        Filter out collisions that should be ignored, based on self._ignore_robot_[object/self]_collisions

        Args:
            collisions (set of 2-tuple): Unique collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths

        Returns:
            set of 2-tuple: Filtered collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths
        """
        # Iterate over all robots
        new_collisions = set()
        for robot, filtered_links, filtered_objs in zip(
                self.robots, self._ignore_robot_self_collisions, self._ignore_robot_object_collisions
        ):
            # Grab all link prim paths owned by the robot
            robot_prims = {link.prim_path for link in robot.links.values()}
            # Loop over all filtered links and compose them
            filtered_link_prims = {link.prim_path for link in filtered_links}
            # Loop over all filtered objects and compose them
            filtered_obj_prims = {link.prim_path for obj in filtered_objs for link in obj.links.values()}
            # Iterate over all collision pairs
            for col_pair in collisions:
                # First check for self_collision -- we check by first subtracting all filtered links from the col_pair
                # set and making sure the length is < 2, and then subtracting all robot_prims and making sure that the
                # length is 0. If both conditions are met, then we know this is a filtered collision!
                col_pair_set = set(col_pair)
                if len(col_pair_set - filtered_link_prims) < 2 and len(col_pair_set - robot_prims) == 0:
                    # Filter this collision
                    continue
                # Check for object filtering -- we check by first subtracting all robot links from the col_pair
                # set and making sure the length is < 2, and then subtracting all filtered_obj_prims and making sure
                # that the length is < 2. If both conditions are met, this means that this was a collision between
                # the robot and a filtered object, so we know this is a filtered collision!
                elif len(col_pair_set - robot_prims) < 2 and len(col_pair_set - filtered_obj_prims) < 2:
                    # Filter this collision
                    continue
                else:
                    # Add this collision
                    new_collisions.add(col_pair)

        return new_collisions

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        info["episode_length"] = self._current_step
        info["collision_step"] = self._collision_step

    def dump_state(self, serialized=False):
        # Default state is from the scene
        return self._scene.dump_state(serialized=serialized)

    def load_state(self, state, serialized=False):
        # Default state is from the scene
        self._scene.load_state(state=state, serialized=serialized)

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: gym.spaces.Dict, dict, np.array, robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        # import pdb
        # pdb.set_trace()
        # If the action is not a dictionary, convert into a dictionary
        if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
            action_dict = OrderedDict()
            idx = 0
            for robot in self.robots:
                action_dim = robot.action_dim
                action_dict[robot.name] = action[idx: idx + action_dim]
                idx += action_dim
        else:
            # Our inputted action is the action dictionary
            action_dict = action

        # Iterate over all robots and apply actions
        for robot in self.robots:
            robot.apply_action(action_dict[robot.name])

        # Run simulation step
        og.sim.step()

        # Grab collisions and store internally
        if gm.ENABLE_GLOBAL_CONTACT_REPORTING:
            collision_objects = self.scene.objects + self.robots
        elif gm.ENABLE_ROBOT_CONTACT_REPORTING:
            collision_objects = self.robots
        else:
            collision_objects = []

        # Update current collisions and corresponding count
        self._current_collisions = self._filter_collisions(collisions=get_collisions(prims=collision_objects))

        # Update the collision count
        self._collision_step += int(len(self._current_collisions) > 0)

        # Grab observations
        obs = self.get_obs()

        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Increment step
        self._current_step += 1

        return obs, reward, done, info

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self._object_randomization_freq is not None:
            if self._current_episode % self._object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self._texture_randomization_freq is not None:
            if self._current_episode % self._texture_randomization_freq == 0:
                og.sim.scene.randomize_texture()

    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episode += 1
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

    # TODO: Match super class signature?
    def reset(self):
        """
        Reset episode.
        """
        # Stop and restart the simulation
        og.sim.stop()
        og.sim.play()

        # Do any domain randomization
        self.randomize_domain()

        # # Move all robots away from the scene since the task will place the robots anyways
        # for robot in self.robots:
        #     robot.set_position(np.array([100.0, 100.0, 100.0]))

        # Reset the task
        self.task.reset(self)

        # Reset internal variables
        self._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.step()

        # Grab and return observations
        obs = self.get_obs()

        # if self.observation_space is not None and not self.observation_space.contains(obs):
        #     # Print out all observations for all robots and task
        #     for robot in self.robots:
        #         for key, value in self.observation_space[robot.name].items():
        #             logging.error(("obs_space", key, value.dtype, value.shape))
        #             logging.error(("obs", key, obs[robot.name][key].dtype, obs[robot.name][key].shape))
        #     for key, value in self.observation_space["task"].items():
        #         logging.error(("obs_space", key, value.dtype, value.shape))
        #         logging.error(("obs", key, obs["task"][key].dtype, obs["task"][key].shape))
        #     raise ValueError("Observation space does not match returned observations!")

        return obs

    # TODO!
    def set_pos_orn_with_z_offset(self, obj, pos, ori=None, offset=None):
        """
        Reset position and orientation for the object @obj with optional offset @offset.

        Args:
            obj (BaseObject): Object to place in the environment
            pos (3-array): Global (x,y,z) location to place the object
            ori (None or 3-array): Optional (r,p,y) orientation when placing the robot. If None, a random orientation
                about the z-axis will be sampled
            offset (None or float): Optional z-offset to place object with. If None, default self._initial_pos_z_offset
                will be used
        """
        # print(f"set ori: {ori}")
        ori = np.array([0, 0, np.random.uniform(0, np.pi * 2)]) if ori is None else ori
        offset = self._initial_pos_z_offset if offset is None else offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*ori), "wxyz"))
        # get the AABB in this orientation
        # lower, _ = obj.states[object_states.AABB].get_value() # TODO!
        lower = np.array([0, 0, pos[2]])
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position(np.array([pos[0], pos[1], stable_z + offset]))
        # Update by taking a sim step
        og.sim._physics_context._step(current_time=og.sim.current_time)
        # self._simulator.app.update()

    def test_valid_position(self, obj, pos, ori=None):
        """
        Test if the object can be placed with no collision.

        Args:
            obj (BaseObject): Object to place in the environment
            pos (3-array): Global (x,y,z) location to place the object
            ori (None or 3-array): Optional (r,p,y) orientation when placing the robot. If None, a random orientation
                about the z-axis will be sampled

        Returns:
            bool: Whether the placed object position is valid
        """
        # Set the position of the object
        self.set_pos_orn_with_z_offset(obj=obj, pos=pos, ori=ori)

        # If we're placing a robot, make sure it's reset and not moving
        if isinstance(obj, BaseRobot):
            obj.reset()
            obj.keep_still()

        # Valid if there are no collisions
        return not self.check_collision(objsA=obj, step_sim=False)

    def check_collision(self, objsA=None, linksA=None, objsB=None, linksB=None, step_sim=False):
        """
        Check whether the given object @objsA or any of @links has collision after one simulator step. If both
        are specified, will take the union of the two.

        Note: This natively checks for collisions with @objsA and @linksA. If @objsB and @linksB are None, any valid
            collision will trigger a True

        Args:
            objsA (None or EntityPrim or list of EntityPrim): If specified, object(s) to check for collision
            linksA (None or RigidPrim or list of RigidPrim): If specified, link(s) to check for collision
            objsB (None or EntityPrim or list of EntityPrim): If specified, object(s) to check for collision with any
                of @objsA or @linksA
            linksB (None or RigidPrim or list of RigidPrim): If specified, link(s) to check for collision with any
                of @objsA or @linksA
            step_sim (bool): Whether to step the simulation first before checking collisions. Default is False

        Returns:
            bool: Whether any of @objsA or @linksA are in collision or not, possibly with @objsB or @linksB if specified
        """
        # Run simulator step and update contacts
        if step_sim:
            self._simulator.app.update()
        collisions = self.update_collisions(filtered=True)

        # Run sanity checks and standardize inputs
        assert objsA is not None or linksA is not None, \
            "Either objsA or linksA must be specified for collision checking!"

        objsA = [] if objsA is None else [objsA] if not isinstance(objsA, Iterable) else objsA
        linksA = [] if linksA is None else [linksA] if not isinstance(linksA, Iterable) else linksA

        # Grab all link prim paths owned by the collision set A
        paths_A = {link.prim_path for obj in objsA for link in obj.links.values()}
        paths_A = paths_A.union({link.prim_path for link in linksA})

        # Determine whether we're checking any collision from collision set A
        check_any_collision = objsB is None and linksB is None

        in_collision = False
        if check_any_collision:
            # Immediately check collisions
            for col_pair in collisions:
                if len(set(col_pair) - paths_A) < 2:
                    in_collision = True
                    break
        else:
            # Grab all link prim paths owned by the collision set B
            objsB = [] if objsB is None else [objsB] if not isinstance(objsB, Iterable) else objsB
            linksB = [] if linksB is None else [linksB] if not isinstance(linksB, Iterable) else linksB
            paths_B = {link.prim_path for obj in objsB for link in obj.links.values()}
            paths_B = paths_B.union({link.prim_path for link in linksB})
            paths_shared = paths_A.intersection(paths_B)
            paths_disjoint = paths_A.union(paths_B) - paths_shared
            is_AB_shared = len(paths_shared) > 0

            # Check collisions specifically between groups A and B
            for col_pair in collisions:
                col_pair = set(col_pair)
                # Two cases -- either paths_A and paths_B overlap or they don't. Process collision checking logic
                # separately for each case
                if is_AB_shared:
                    # Two cases for valid collision: there is a shared collision body in this pair or there isn't.
                    # Process separately in each case
                    col_pair_no_shared = col_pair - paths_shared
                    if len(col_pair_no_shared) < 2:
                        # Make sure this set minus the disjoint set results in empty col_pair remaining -- this means
                        # a valid pair combo was found
                        if len(col_pair_no_shared - paths_disjoint) == 0:
                            in_collision = True
                            break
                    else:
                        # Make sure A and B sets each have an entry in the col pair for a valid collision
                        if len(col_pair - paths_A) == 1 and len(col_pair - paths_B) == 1:
                            in_collision = True
                            break
                else:
                    # Make sure A and B sets each have an entry in the col pair for a valid collision
                    if len(col_pair - paths_A) == 1 and len(col_pair - paths_B) == 1:
                        in_collision = True
                        break

        # Only going into this if it is for logging --> efficiency
        if logging.root.level <= logging.DEBUG:
            for item in collisions:
                logging.debug("linkA:{}, linkB:{}".format(item[0], item[1]))

        return in_collision

    # TODO!! Wait until omni dev team has an answer
    def update_collisions(self, filtered=True):
        """
        Grab collisions that occurred during the most recent physics timestep

        Args:
            filtered (bool): Whether to filter the raw collisions or not based on
                self._ignore_robot_[object/self]_collisions

        Returns:
            set of 2-tuple: Unique collision pairs occurring in the simulation at the current timestep, represented
                by their prim_paths
        """
        # Grab collisions based on the status of our contact reporting macro flag
        if m.ENABLE_GLOBAL_CONTACT_REPORTING:
            collisions = {(c.body0, c.body1)
                          for obj_group in (self.scene.objects, self.robots)
                          for obj in obj_group
                          for c in obj.contact_list()}
        elif m.ENABLE_ROBOT_CONTACT_REPORTING:
            collisions = {(c.body0, c.body1) for robot in self.robots for c in robot.contact_list()}
        else:
            collisions = set()
        # print(f"collisions: {collisions}")
        self._current_collisions = self._filter_collisions(collisions) if filtered else collisions
        # print(f"filtered collisions: {self._current_collisions}")
        return self._current_collisions

    def _filter_collisions(self, collisions):
        """
        Filter out collisions that should be ignored, based on self._ignore_robot_[object/self]_collisions

        Args:
            collisions (set of 2-tuple): Unique collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths

        Returns:
            set of 2-tuple: Filtered collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths
        """
        # Iterate over all robots
        new_collisions = set()
        for robot, filtered_links, filtered_objs in zip(
                self.robots, self._ignore_robot_self_collisions, self._ignore_robot_object_collisions
        ):
            # Grab all link prim paths owned by the robot
            robot_prims = {link.prim_path for link in robot.links.values()}
            # Loop over all filtered links and compose them
            filtered_link_prims = {link.prim_path for link in filtered_links}
            # Loop over all filtered objects and compose them
            filtered_obj_prims = {link.prim_path for obj in filtered_objs for link in obj.links.values()}
            # Get union of robot and obj prims
            filtered_robot_obj_prims = robot_prims.union(filtered_obj_prims)
            # Iterate over all collision pairs
            for col_pair in collisions:
                # First check for self_collision -- we check by first subtracting all filtered links from the col_pair
                # set and making sure the length is < 2, and then subtracting all robot_prims and making sure that the
                # length is 0. If both conditions are met, then we know this is a filtered collision!
                col_pair_set = set(col_pair)
                if len(col_pair_set - filtered_link_prims) < 2 and len(col_pair_set - robot_prims) == 0:
                    # Filter this collision
                    continue
                # Check for object filtering -- we check by first subtracting all robot links from the col_pair
                # set and making sure the length is < 2, and then subtracting all filtered_obj_prims and making sure
                # that the length is < 2. If both conditions are met, this means that this was a collision between
                # the robot and a filtered object, so we know this is a filtered collision!
                elif len(col_pair_set - robot_prims) < 2 and len(col_pair_set - filtered_obj_prims) < 2:
                    # Filter this collision
                    continue
                else:
                    # Add this collision
                    new_collisions.add(col_pair)

        return new_collisions

    @property
    def episode_steps(self):
        """
        Returns:
            int: Current number of steps in episode
        """
        return self._current_step

    @property
    def episode_collisions(self):
        """
        Returns:
            int: Total number of collisions in current episode
        """
        return self._collision_step

    @property
    def current_collisions(self):
        """
        Returns:
            set of 2-tuple: Cached collision pairs from the last time self.update_collisions() was called
        """
        return self._current_collisions

    @property
    def initial_pos_z_offset(self):
        """
        Returns:
            float: how high to offset object placement to test valid pose & account for one action step of dropping
        """
        return self._initial_pos_z_offset

    @property
    def task(self):
        """
        Returns:
            BaseTask: Active task instance
        """
        return self._task

    @property
    def scene(self):
        """
        Returns:
            Scene: Active scene in this environment
        """
        return self._scene

    @property
    def robots(self):
        """
        Returns:
            list of BaseRobot: Robots in the current scene
        """
        return self._scene.robots

    @property
    def env_config(self):
        """
        Returns:
            dict: Environment-specific configuration kwargs
        """
        return self.config["env"]

    @property
    def render_config(self):
        """
        Returns:
            dict: Render-specific configuration kwargs
        """
        return self.config["render"]

    @property
    def scene_config(self):
        """
        Returns:
            dict: Scene-specific configuration kwargs
        """
        print(self.config["scene"])
        return self.config["scene"]

    @property
    def robots_config(self):
        """
        Returns:
            dict: Robot-specific configuration kwargs
        """
        return self.config["robots"]

    @property
    def task_config(self):
        """
        Returns:
            dict: Task-specific configuration kwargs
        """
        return self.config["task"]

    @property
    def default_config(self):
        """
        Returns:
            dict: Default configuration for this environment. May not be fully specified (i.e.: still requires @config
                to be specified during environment creation)
        """
        return {
            # Environment kwargs
            "env": {
                "initial_pos_z_offset": 0.1,
                "texture_randomization_freq": None,
                "object_randomization_freq": None,
            },

            # Rendering kwargs
            "render": {
                "viewer_width": 1280,
                "viewer_height": 720,
                "vertical_fov": 90,
                # "optimized_renderer": True,
            },

            # Scene kwargs
            "scene": {
                # Traversibility map kwargs
                "waypoint_resolution": 0.2,
                "num_waypoints": 10,
                "build_graph": False,
                "trav_map_resolution": 0.1,
                "trav_map_erosion": 2,
                "trav_map_with_objects": True,
                "usd_file": None,
                "usd_path": None,
            },

            # Robot kwargs
            "robots": [], # no robots by default

            # Task kwargs
            "task": {
                "type": "DummyTask",

                # If we're using a BehaviorTask
                "activity_definition_id": 0,
                "activity_instance_id": 0,
            }
        }
