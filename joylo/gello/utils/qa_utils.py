import omnigibson as og
from omnigibson.envs import EnvMetric
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from omnigibson.utils.backend_utils import _compute_backend as cb
from gello.robots.sim_robot.og_teleop_utils import GHOST_APPEAR_THRESHOLD
import torch as th
import operator


class MotionMetric(EnvMetric):
    def __init__(self, step_dt):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
        """
        self.step_dt = step_dt

        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            # Record velocity (we'll derive accel -> jerk at the end of the episode)
            step_metrics[f"robot{i}::pos"] = robot.get_joint_positions()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        episode_metrics = dict()
        for pos_key, positions in episode_info.items():
            positions = th.stack(positions, dim=0)
            vels = (positions[1:] - positions[:-1]) / self.step_dt
            accs = (vels[1:] - vels[:-1]) / self.step_dt
            jerks = (accs[1:] - accs[:-1]) / self.step_dt
            episode_metrics[f"{pos_key}::vel_avg"] = vels.mean(dim=0)
            episode_metrics[f"{pos_key}::acc_avg"] = accs.mean(dim=0)
            episode_metrics[f"{pos_key}::jerk_avg"] = jerks.mean(dim=0)
            episode_metrics[f"{pos_key}::vel_std"] = vels.std(dim=0)
            episode_metrics[f"{pos_key}::acc_std"] = accs.std(dim=0)
            episode_metrics[f"{pos_key}::jerk_std"] = jerks.std(dim=0)
            episode_metrics[f"{pos_key}::vel_max"] = vels.max().item()
            episode_metrics[f"{pos_key}::acc_max"] = accs.max().item()
            episode_metrics[f"{pos_key}::jerk_max"] = jerks.max().item()

        return episode_metrics


class CollisionMetric(EnvMetric):
    def __init__(self):
        self.checks = dict()
        super().__init__()

    def add_check(self, name, check):
        """
        Adds a collision check to this metric, which can be queried by @name

        Args:
            name (str): name of the check
            check (function): Collision checker function, with the following signature:

                def check(env: Environment) -> bool

                which should return True if there is collision, else False
        """
        self.checks[name] = check

    def remove_check(self, name):
        """
        Removes check with corresponding @name

        Args:
            name (str): name of the check to remove
        """
        self.checks.pop(name)

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for name, check in self.checks.items():
            step_metrics[f"{name}"] = check(env)
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Compute any collisions from this step
        episode_metrics = dict()
        for name, collisions in episode_info.items():
            collisions = th.tensor(collisions)
            episode_metrics[f"{name}::n_collision"] = collisions.sum().item()

        return episode_metrics


class TaskSuccessMetric(EnvMetric):

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        return {"done": terminated and not truncated}

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        return {"success": th.any(th.tensor(episode_info["done"])).item()}


class GhostHandAppearanceMetric(EnvMetric):

    @classmethod
    def is_compatible(cls, env):
        valid = super().is_compatible(env=env)
        if valid:
            # We must be using a binary / smooth gripper controller for each robot to ensure that we can
            # infer un/grasping intent
            for robot in env.robots:
                for arm in robot.arm_names:
                    gripper_controller = robot.controllers[f"gripper_{arm}"]
                    is_1d = gripper_controller.command_dim == 1
                    is_normalized = (th.all(cb.to_torch(gripper_controller.command_input_limits[0]) == -1.0).item() and
                                     th.all(cb.to_torch(gripper_controller.command_input_limits[1]) == 1.0).item())
                    valid = is_1d and is_normalized
                    if not valid:
                        break
                if not valid:
                    break
        return valid

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            robot_qpos = robot.get_joint_positions(normalized=False)
            gripper_action_idxs = robot.gripper_action_idx
            for arm in robot.arm_names:
                active = th.max(th.abs(
                        robot_qpos[robot.arm_control_idx[arm]] - action[robot.arm_action_idx[arm]]
                )).item() > GHOST_APPEAR_THRESHOLD
                gripper_controller = robot.controllers[f"gripper_{arm}"]
                step_metrics[f"robot{i}::arm_{arm}::active"] = active
                op = operator.lt if gripper_controller._inverted else operator.ge
                step_metrics[f"robot{i}::arm_{arm}::open_cmd"] = th.all(op(gripper_action_idxs[arm], 0.0)).item()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Aggregate number of steps the ghost hands have appeared per-arm, and whether the robot was releasing a grasp
        # during that time
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                active = th.tensor(episode_info[f"{pf}::active"])
                open_cmd = th.tensor(episode_info[f"{pf}::open_cmd"])
                ungrasping = open_cmd[1:] & ~open_cmd[:-1]
                episode_metrics[f"{pf}::n_steps"] = active.sum().item()
                episode_metrics[f"{pf}::n_steps_while_ungrasping"] = (active[1:] & ungrasping).sum().item()
        return episode_metrics


class ProlongedPauseMetric(MotionMetric):

    def __init__(self, step_dt, vel_threshold=0.001):
        """
        Args:
            step_dt (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
            vel_threshold (float): Per-joint vel threshold for determining whether there's any motion occurring
                at a given step
        """
        self.vel_threshold = vel_threshold
        super().__init__(step_dt=step_dt)

    def _compute_episode_metrics(self, env, episode_info):
        # Derive velocities, then count consecutive steps that contain values greater than our threshold
        episode_metrics = dict()
        for pos_key, positions in episode_info.items():
            positions = th.stack(positions, dim=0)
            vels = (positions[1:] - positions[:-1]) / self.step_dt
            in_motions = th.any(th.abs(vels) > self.vel_threshold, dim=-1)
            max_pause_length = 0
            current_pause_length = 0
            for in_motion in in_motions:
                if not in_motion.item():
                    current_pause_length += 1
                    if current_pause_length > max_pause_length:
                        max_pause_length = current_pause_length
                else:
                    current_pause_length = 0
            episode_metrics[f"{pos_key}::max_pause_length"] = max_pause_length

        return episode_metrics


class FailedGraspMetric(EnvMetric):

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        step_metrics = dict()
        for i, robot in enumerate(env.robots):
            # Record whether fingers are closed (values ~ 0) -- this implies a failed grasp
            for arm in robot.arm_names:
                step_metrics[f"robot{i}::arm_{arm}::fingers_closed"] = th.allclose(robot.get_joint_positions()[robot.gripper_control_idx[arm]], th.zeros(2), atol=1e-4).item()
        return step_metrics

    def _compute_episode_metrics(self, env, episode_info):
        # Compute all fingers_closed upticks to get total failed grasp count
        episode_metrics = dict()
        for i, robot in enumerate(env.robots):
            for arm in robot.arm_names:
                pf = f"robot{i}::arm_{arm}"
                fingers_closed = th.tensor(episode_info[f"{pf}::fingers_closed"])
                fingers_closed_transition = fingers_closed[1:] & ~fingers_closed[:-1]
                episode_metrics[f"{pf}::failed_grasp_count"] = fingers_closed_transition.sum().item()

        return episode_metrics


def check_robot_self_collision(env):
    # TODO: What about gripper finger self collision?
    for robot in env.robots:
        link_paths = robot.link_prim_paths
        if RigidContactAPI.in_contact(link_paths, link_paths):
            return True
    return False


def check_robot_base_nonarm_nonfloor_collision(env):
    # TODO: How to check for wall collisions? They're kinematic only
    # # One solution: Make them non-kinematic only during QA checking
    # floor_link_paths = []
    # for structure_category in STRUCTURE_CATEGORIES:
    #     for structure in env.scene.object_registry("category", structure_category):
    #         floor_link_paths += structure.link_prim_paths
    # floor_link_col_idxs = {RigidContactAPI.get_body_col_idx(link_path) for link_path in floor_link_paths}

    for robot in env.robots:
        robot_link_paths = set(robot.link_prim_paths)
        for arm in robot.arm_names:
            robot_link_paths -= set(link.prim_path for link in robot.arm_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.gripper_links[arm])
            robot_link_paths -= set(link.prim_path for link in robot.finger_links[arm])
    robot_link_idxs = [RigidContactAPI.get_body_col_idx(link_path)[1] for link_path in robot_link_paths]
    robot_contacts = RigidContactAPI.get_all_impulses(env.scene.idx)[robot_link_idxs]

    return th.any(robot_contacts).item()

