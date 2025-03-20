import os
import pdb

import torch
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import BimanualKeyboardRobotController, choose_from_options

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False


from omnigibson.utils.control_utils import orientation_error

def generate_waypoint_sequence(env, robot):
    """
    Returns:
        dict: a dictionary of waypoints for the robot to follow
    """

    # the waypoints are pre-recorded with teleoperation
    waypoints = {
        "arm_left": (
            (None, None),  # "pre_grasp_pos"
            # (torch.Tensor([0.6114, -0.2484,  0.7222]), None), # 'grasp_pos'
            # (torch.Tensor([0.5330, 0.0707, 0.7473]), None), # 'pre_coordinate_pose'
            # (torch.Tensor([ 0.5166, -0.3180,  0.7871]), None), # 'coordinate_pose'Component: arm_left, Info: {'name': 'InverseKinematicsController', 'start_idx': 5, 'dofs': tensor([ 6,  7, 10, 13, 15, 17, 19, 21], dtype=torch.int32), 'command_dim': 6}
        ),
        "arm_right": (
            (
                torch.Tensor([0.5982, -0.2320, 0.8175]),
                torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
            ),  # "pre_grasp_pos"
            (torch.Tensor([0.5982, -0.2320, 0.7175]), torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509])),  # 'grasp_pos'
            # (torch.Tensor([0.4982, -0.1320,  0.8175]), torch.Tensor([-0.6253, -0.6140,  0.3300, -0.3509])), # 'pre_coordinate_pose'
            # (torch.Tensor([ 0.5166, -0.3180,  0.7871]), None), # 'coordinate_pose'
        ),
    }
    return waypoints


def get_action_linear(start, end, steps=20):
    pdb.set_trace()
    # interpolate between the two multi-dimensional array
    position_sequence = torch.stack([torch.linspace(start[i], end[i], steps) for i in range(len(start))], dim=1)
    # action sequence is the difference between the two array
    action_sequence = position_sequence[1:] - position_sequence[:-1]
    return action_sequence


def generate_action_sequence(key_pos_dict):
    # input: the environment
    # output: a list of actions

    # left arm: pregrasp the paper_cup_1, grasp the cup, lift the cup to target position

    # get left arm end effector position
    left_eef_pos = robot.get_eef_position("left")
    lef_eef_orientation = robot.get_eef_orientation("left")

    right_eef_pos = robot.get_eef_position("right")
    right_eef_orientation = robot.get_eef_orientation("right")

    name = "right"
    waypoints = key_pos_dict["right"]
    num_stages = len(waypoints)

    target_pos = waypoints["pre_grasp_pos"]
    cur_pos = robot.get_eef_position("right")
    action_sequence = get_action_linear(cur_pos, target_pos, steps=3)
    return action_sequence


def _empty_action(robot):
    import torch as th

    """
    Get a no-op action that allows us to run simulation without changing robot configuration.

    Returns:
        th.tensor or None: Action array for one step for the robot to do nothing
    """
    action = th.zeros(robot.action_dim)
    for name, controller in robot._controllers.items():
        action_idx = robot.controller_action_idx[name]
        no_op_action = controller.compute_no_op_action(robot.get_control_dict())
        action[action_idx] = no_op_action
    return action


def format_action():
    base_action = torch.Tensor([0.0, 0.0, 0.0])
    camera_action = torch.Tensor([0.0, 0.0])
    arm_left = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arm_right = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripper_action_left = torch.Tensor([0.0])
    gripper_action_right = torch.Tensor([0.0])
    # concatenate all actions
    action = torch.cat([base_action, camera_action, arm_left, gripper_action_left, arm_right, gripper_action_right])
    return action


def send_to_target_pose(robot, name, target_position, target_orientation):
    # name = 'arm_left'
    ik_controller = robot.controllers[name]

    # target_position = torch.Tensor([0.5714, 0.15,  0.7222])
    # target_orientation=torch.Tensor([-0.6877,  0.6333,  0.2396,  0.2618])

    # torch.Tensor([0.5814, 0.15,  0.7222]), torch.Tensor([-0.6877,  0.6333,  0.2396,  0.2618])

    # goal_pos = ik_controller._goal
    # target_position  = ik_controller._goal['target_pos'] + th.tensor([-0.1,0.0,0.1])
    # target_orientation = ik_contrself.arm_command_start_idxoller._goal['target_quat']

    # target_orientation = T.euler2quat( T.quat2euler(ik_controller._goal['target_quat']) + th.tensor([0.0,0.0,1]))

    ik_controller._goal = {"target_pos": target_position, "target_quat": target_orientation}
    for _ in range(50):
        og.sim.step()


def get_eef_pos_orn(robot, name):
    # name sample from 'left' or 'right'
    current_pos = robot.get_eef_position(name)
    current_orn = robot.get_eef_orientation(name)
    return torch.cat([current_pos, current_orn])


# def change_friction(object, target_friction):
#     # maintain env state
#     state = og.sim.dump_state()
#     og.sim.stop()

#     pdb.set_trace()
#     # apply gripper material
#     gripper_mat = lazy.omni.isaac.core.materials.PhysicsMaterial(
#         prim_path=f"{object.prim_path}/{object}_mat",
#         name=f"{object}_mat",
#         static_friction=target_friction,
#         dynamic_friction=target_friction,
#         restitution=None,
#     )
#     for arm, links in object.finger_links.items():
#         for link in links:
#             for msh in link.collision_meshes.values():
#                 msh.apply_physics_material(gripper_mat)

#     og.sim.play()
#     og.sim.load_state(state)
#     for _ in range(10): og.sim.step()


def close_gripper(env, robot, name="right"):
    action = _empty_action(robot)
    if name == "right":
        action[-1] = -1.0
    if name == 'left':
        action[-8] = -1.0
    for _ in range(10):
        env.step(action=action)
    print(f"gripper {name} is closed")


def open_gripper(env, robot, name="left"):
    action = _empty_action(robot)
    if name == "right":
        action[-1] = 1.0
    if name == 'left':
        action[-8] = 1.0
    for _ in range(10):
        env.step(action=action)
    print(f"gripper {name} is opened")


# before merge the tiago config
def move_to_waypoints_old(waypoints, env, robot):
    num_waypoints = max(len(waypoints["arm_right"]), len(waypoints["arm_left"]))

    l = 0
    r = 0
    counter_l = 0
    counter_r = 0

    while True:

        l_waypoint = waypoints["arm_left"][min(l, len(waypoints["arm_left"]) - 1)]
        r_waypoint = waypoints["arm_right"][min(r, len(waypoints["arm_right"]) - 1)]
        current_waypoint = {"arm_left": l_waypoint, "arm_right": r_waypoint}

        l_reached_target = False
        r_reached_target = False

        while not l_reached_target and not r_reached_target:
            action, updated_waypoint = generate_action(robot, current_waypoint)
            print('left gripper action', action[-8])
            print('right gripper action', action[-1])

            if action[-8] == 1:
                pdb.set_trace()
            env.step(action=action)

            r_robot_eef_pos = robot.get_eef_position("right")
            r_robot_eef_orn = robot.get_eef_orientation("right")
            l_robot_eef_pos = robot.get_eef_position("left")
            l_robot_eef_orn = robot.get_eef_orientation("left")

            # Tweak this tolerance

            l_reached_target = (
                th.isclose(l_robot_eef_pos, updated_waypoint["arm_left"][0], atol=0.01).all()
                and th.isclose(l_robot_eef_orn, updated_waypoint["arm_left"][1], atol=0.01).all()
            )
            r_reached_target = (
                th.isclose(r_robot_eef_pos, updated_waypoint["arm_right"][0], atol=0.01).all()
                and th.isclose(r_robot_eef_orn, updated_waypoint["arm_right"][1], atol=0.01).all()
            )

            counter_l += 1
            counter_r += 1
            print("counter_l:", counter_l, "counter_r:", counter_r)

            if counter_r > 100:
                r_reached_target = True

            if counter_l > 100:
                l_reached_target = True

        if l_reached_target:
            if l_waypoint[-1] == -1:
                close_gripper(env, robot, "left")
            if l_waypoint[-1] == 1:
                pdb.set_trace()
                open_gripper(env, robot, "left")
            counter_l = 0
        if r_reached_target:
            if r_waypoint[-1] == -1:
                close_gripper(env, robot, "right")
            if r_waypoint[-1] == 1:
                open_gripper(env, robot, "right")
            counter_r = 0

        if l < len(waypoints["arm_left"]) and l_reached_target:
            l += 1
            print("left arm reached target", "left:", l, "right:", r)
        if r < len(waypoints["arm_right"]) and r_reached_target:
            r += 1
            print("right arm reached target", "left:", l, "right:", r)

        if l == len(waypoints["arm_left"]) and r == len(waypoints["arm_right"]):
            pdb.set_trace()
            break


def generate_action(robot, arm_targets):
    """
    Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

    Returns:
        th.tensor or None: Action array for one step for the robot to do nothing
    """
    action = th.zeros(robot.action_dim)
    for name, controller in robot._controllers.items():
        # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
        if name in arm_targets:
            arm = name.replace("arm_", "")
            # target_pos, target_orn_axisangle = arm_targets[name]
            # print()
            # print('arm_targets', arm_targets)
            # print('name', name)
            target_pos, target_orn, gripper_state = arm_targets[name]
            current_pos = robot.get_eef_position(arm)
            current_orn = robot.get_eef_orientation(arm)
            if target_orn is None:
                target_orn = current_orn
            if target_pos is None:
                target_pos = current_pos
            arm_targets[name] = (target_pos, target_orn, gripper_state)

            delta_pos = target_pos - current_pos
            # delta_orn = orientation_error(T.quat2mat(T.axisangle2quat(target_orn_axisangle)), T.quat2mat(current_orn))
            delta_orn = orientation_error(T.quat2mat(target_orn), T.quat2mat(current_orn))
            partial_action = th.cat((delta_pos, delta_orn))
        else:
            partial_action = controller.compute_no_op_action(robot.get_control_dict())
        action_idx = robot.controller_action_idx[name]
        action[action_idx] = partial_action

        # set the gripper no operation action to 0
        action[-8] = 0
        action[-1] = 0
    return action, arm_targets


# after merge the new tiago config 
def move_to_waypoints(waypoints, env, robot):
    num_waypoints = max(len(waypoints["arm_right"]), len(waypoints["arm_left"]))

    l = 0
    r = 0
    counter_l = 0
    counter_r = 0

    while True:

        l_waypoint = waypoints["arm_left"][min(l, len(waypoints["arm_left"]) - 1)]
        r_waypoint = waypoints["arm_right"][min(r, len(waypoints["arm_right"]) - 1)]
        current_waypoint = {"arm_left": l_waypoint, "arm_right": r_waypoint}

        l_reached_target = False
        r_reached_target = False

        while not l_reached_target and not r_reached_target:
            action, updated_waypoint = generate_action(robot, current_waypoint)
            
            if l_waypoint[-1] == -1: 
                action[-8] = -1
            else:
                action[-8] = 1
            if r_waypoint[-1] == -1:
                action[-1] = -1
            else:
                action[-1] = 1

            # print('left gripper action', action[-8])
            # print('right gripper action', action[-1])
            print('base, trunk and camera action', action[:6])
            print('arm and gripper actions', action[6:])
            env.step(action=action)

            r_robot_eef_pos = robot.get_eef_position("right")
            r_robot_eef_orn = robot.get_eef_orientation("right")
            l_robot_eef_pos = robot.get_eef_position("left")
            l_robot_eef_orn = robot.get_eef_orientation("left")

            # Tweak this tolerance

            l_reached_target = (
                th.isclose(l_robot_eef_pos, updated_waypoint["arm_left"][0], atol=0.01).all()
                and th.isclose(l_robot_eef_orn, updated_waypoint["arm_left"][1], atol=0.01).all()
            )
            r_reached_target = (
                th.isclose(r_robot_eef_pos, updated_waypoint["arm_right"][0], atol=0.01).all()
                and th.isclose(r_robot_eef_orn, updated_waypoint["arm_right"][1], atol=0.01).all()
            )

            counter_l += 1
            counter_r += 1
            print("counter_l:", counter_l, "counter_r:", counter_r)

            if counter_r > 100:
                r_reached_target = True

            if counter_l > 100:
                l_reached_target = True

        if l_reached_target:
            counter_l = 0
        if r_reached_target:
            counter_r = 0

        if l < len(waypoints["arm_left"]) and l_reached_target:
            l += 1
            print("left arm reached target", "left:", l, "right:", r)
        if r < len(waypoints["arm_right"]) and r_reached_target:
            r += 1
            print("right arm reached target", "left:", l, "right:", r)

        if l == len(waypoints["arm_left"]) and r == len(waypoints["arm_right"]):
            pdb.set_trace()
            break


def main():
    # config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    config_filename = os.path.join(og.example_config_path, "tiago_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # activity_name = "test_cabinet"
    activity_name = "test_tiago_cup"
    cfg["task"]["activity_name"] = activity_name
    cfg["task"]["online_object_sampling"] = False
    cfg["env"]["flatten_obs_space"] = True
    cfg["env"]["action_frequency"] = 30
    cfg["env"]["rendering_frequency"] = 30
    cfg["env"]["physics_frequency"] = 120
    cfg["robots"][0]["default_reset_mode"] = "untuck"

    collect_hdf5_path = f"{activity_name}.hdf5"

    # Load the environment
    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        optimize_sim=False,
    )
    robot = env.robots[0]

    state = og.sim.dump_state()
    og.sim.stop()
    coffee_cup = env.scene.object_registry("name", "coffee_cup")
    coffee_cup.links["base_link"].density = 100
    # coffee_cup.links['base_link'].friction = 0.01 # friction is not in the link object
    paper_cup = env.scene.object_registry("name", "paper_cup")
    paper_cup.links["base_link"].density = 100
    og.sim.play()
    og.sim.load_state(state)
    for _ in range(10):
        og.sim.step()

    # Create teleop controller
    action_generator = BimanualKeyboardRobotController(robot=robot)

    def start_teleop(env=env, robot=robot, action_generator=action_generator):
        for _ in range(500):
            action = action_generator.get_teleop_action_bimanual()
            next_obs, reward, terminated, truncated, info = env.step(action=action)
        print("arm_left:", get_eef_pos_orn(robot, "left"))
        print("arm_right:", get_eef_pos_orn(robot, "right"))

    def render(steps):
        for _ in range(steps):
            og.sim.render()

    pdb.set_trace()

    # pre_grasp_pos_left = paper_cup.get_position_orientation()[0] - torch.Tensor([0.1, 0, 0])
    # pre_grasp_ori_left = paper_cup.get_position_orientation()[1]
    # # concat the position and orientation
    # pre_grasp_left = torch.cat([pre_grasp_pos_left, pre_grasp_ori_left])

    # send_to_target_pose(robot, 'arm_left', torch.Tensor([0.6, 0.2228, 0.7436]), torch.Tensor([-0.6877,  0.6333,  0.2396,  0.2618]))

    # before merge the tiago config
    waypoints_list_old = [
        {
            "arm_left": (
                (torch.Tensor([0.6, 0.2228, 0.8136 - 0.2]), torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]), 0),
                (
                    torch.Tensor([0.6, 0.2228, 0.7436 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    0,
                ),  # "pre_grasp_pos"
                (
                    torch.Tensor([0.6, 0.15, 0.7022 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    -1,
                ),  # 'grasp_pos'
                (
                    torch.Tensor([0.6, 0.15, 0.8022 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    0,
                ),  # 'pre_coordinate_pose'
            ),
            "arm_right": (
                (
                    torch.Tensor([0.5982, -0.2320, 0.8175 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # "pre_grasp_pos"
                (
                    torch.Tensor([0.5982, -0.2320, 0.7175 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    -1,
                ),  # 'grasp_pos'
                (
                    torch.Tensor([0.5982, -0.220, 0.7675 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'pre_coordinate_pose'
                (
                    torch.Tensor([0.4966, -0.120, 0.7175 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    1,
                ),  # 'coordinate_pose'
                (
                    torch.Tensor([0.4966, -0.120, 0.8075 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'post place'
                (
                    torch.Tensor([0.5066, -0.300, 0.8075 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'post place'
            ),
        },
        {
            "arm_left": (
                (
                    torch.Tensor([0.6, -0.05, 0.8022 - 0.2]),
                    torch.Tensor([-0.5733, 0.4663, 0.4866, 0.4659]),
                    0,
                ),  # 'coordinate_pose'
                (
                    torch.Tensor([0.6, 0.10, 0.8022 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    0,
                ),  # 'grasp_pos'
                (
                    torch.Tensor([0.6, 0.10, 0.7022 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    1,
                ),  # 'grasp_pos'
                # (torch.Tensor([0.6, 0.2228, 0.8136]), torch.Tensor([-0.6877,  0.6333,  0.2396,  0.2618]), 0)
            ),
            "arm_right": ((None, None, 0),),
        },
    ]

    # after merge the tiago config
    waypoints_list = [
        {
            "arm_left": (
                (torch.Tensor([0.6, 0.28, 0.8 - 0.2]), torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]), 0),
                (
                    torch.Tensor([0.6, 0.28, 0.75 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    0,
                ),  # "pre_grasp_pos"
                (
                    torch.Tensor([0.6, 0.17, 0.72 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    0,
                ),  # 'grasp_pos'
                (
                    torch.Tensor([0.6, 0.17, 0.72 - 0.2]),
                    torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]),
                    -1,
                ),  # 'pre_coordinate_pose'
            ),
            "arm_right": (
                (
                    torch.Tensor([0.5982, -0.14, 0.9 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # "pre_grasp_pos"
                (
                    torch.Tensor([0.5982, -0.14, 0.7 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'grasp_pos'
                (torch.Tensor([0.5982, -0.14, 0.75 - 0.2]), torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]), -1),
                (
                    torch.Tensor([0.4, -0.15, 0.75 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    -1,
                ),  # 'pre_coordinate_pose'
                (
                    torch.Tensor([0.4, -0.0, 0.71 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    -1,
                ),  # 'coordinate_pose'
                (
                    torch.Tensor([0.44, -0.1, 0.85 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'post place'
                (
                    torch.Tensor([0.45, -0.200, 0.8075 - 0.2]),
                    torch.Tensor([-0.6253, -0.6140, 0.3300, -0.3509]),
                    0,
                ),  # 'post place'
            ),
        },
        {
            "arm_left": (
                (torch.Tensor([0.5593, 0.17, 0.68]), torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]), -1),
                (torch.Tensor([0.5593, 0.05, 0.68]), torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]), -1),
                (torch.Tensor([0.5593, 0.05, 0.65]), torch.Tensor([-0.6877, 0.6333, 0.2396, 0.2618]), 0),
            ),
            "arm_right": ((None, None, 0),),
        },
    ]

    for waypoint in waypoints_list:
        move_to_waypoints(waypoint, env, robot)

    pdb.set_trace()
    print("now the waypoint is reached")

    print("Data saved")
    env.save_data()

    og.shutdown()


if __name__ == "__main__":
    main()