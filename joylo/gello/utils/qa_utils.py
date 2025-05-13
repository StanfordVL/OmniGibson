import omnigibson as og
from omnigibson.envs import EnvMetric
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from gello.robots.sim_robot.og_teleop_utils import GHOST_APPEAR_THRESHOLD
import torch as th


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
            episode_metrics[f"{pos_key}::avg_vel"] = vels.mean(dim=0)
            episode_metrics[f"{pos_key}::avg_acc"] = accs.mean(dim=0)
            episode_metrics[f"{pos_key}::avg_jerk"] = jerks.mean(dim=0)
            episode_metrics[f"{pos_key}::max_vel"] = vels.max().item()
            episode_metrics[f"{pos_key}::max_acc"] = accs.max().item()
            episode_metrics[f"{pos_key}::max_jerk"] = jerks.max().item()

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

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        active = False
        for robot in env.robots:
            robot_qpos = robot.get_joint_positions(normalized=False)
            for arm in robot.arm_names:
                if th.max(th.abs(
                        robot_qpos[robot.arm_control_idx[arm]] - action[robot.arm_action_idx[arm]]
                )).item() > GHOST_APPEAR_THRESHOLD:
                    active = True
                    break
            if active:
                break
        return {"active": active}

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        return {"n_steps_active": th.tensor(episode_info["active"]).sum().item()}


class ProlongedPauseMetric(EnvMetric):

    def __init__(self, motion_threshold=0.01):
        """
        Args:
            motion_threshold (float): Amount of time between steps, used to differentiate from pos -> vel -> acc -> jerk
        """
        self.motion_threshold = motion_threshold

        super().__init__()

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        # Record whether task is done (terminated is true but not truncated)
        active = False
        for robot in env.robots:
            robot_qpos = robot.get_joint_positions(normalized=False)
            for arm in robot.arm_names:
                if th.max(th.abs(
                        robot_qpos[robot.arm_control_idx[arm]] - action[robot.arm_action_idx[arm]]
                )).item() > GHOST_APPEAR_THRESHOLD:
                    active = True
                    break
            if active:
                break
        return {"active": active}

    def _compute_episode_metrics(self, env, episode_info):
        # Derive acceleration -> jerk based on the recorded velocities
        return {"n_steps_active": th.tensor(episode_info["active"]).sum().item()}



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

