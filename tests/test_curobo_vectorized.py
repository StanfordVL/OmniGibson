import gc
import math
import os
import yaml
from collections import defaultdict

import torch as th
import numpy as np

import omnigibson as og
from omnigibson.action_primitives.curobo_vectorized import CuRoboMotionGenerator
from omnigibson.macros import gm
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
import omnigibson.utils.transform_utils as T



def test_curobo():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES

    # Create env
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 300,
        },
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "PrimitiveObject",
                "name": "obj0",
                "primitive_type": "Cube",
                "scale": [0.4, 0.4, 0.4],
                "fixed_base": True,
                "position": [0.5, -0.1, 0.2],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "eef_marker_0",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [1, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "eef_marker_1",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [0, 1, 0, 1],
            },
        ],
        "robots": [],
    }

    robot_cfgs = [
        # {
        #     "type": "FrankaPanda",
        #     "obs_modalities": "rgb",
        #     "position": [0.7, -0.55, 0.0],
        #     "orientation": [0, 0, 0.707, 0.707],
        #     "self_collisions": True,
        #     "action_normalize": False,
        #     "controller_config": {
        #         "arm_0": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #         "gripper_0": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #     },
        # },
        # {
        #     "type": "R1",
        #     "obs_modalities": "rgb",
        #     "position": [0.7, -0.7, 0.056],
        #     "orientation": [0, 0, 0.707, 0.707],
        #     "self_collisions": True,
        #     "action_normalize": False,
        #     "controller_config": {
        #         "base": {
        #             "name": "HolonomicBaseJointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_impedances": False,
        #         },
        #         "trunk": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #         "arm_left": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #         "arm_right": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #         "gripper_left": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #         "gripper_right": {
        #             "name": "JointController",
        #             "motor_type": "position",
        #             "command_input_limits": None,
        #             "use_delta_commands": False,
        #             "use_impedances": False,
        #         },
        #     },
        # },
        {
            "type": "Tiago",
            "obs_modalities": "rgb",
            "position": [0.7, -0.85, 0],
            "orientation": [0, 0, 0.707, 0.707],
            "self_collisions": True,
            "action_normalize": False,
            "controller_config": {
                "base": {
                    "name": "HolonomicBaseJointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_impedances": False,
                },
                "trunk": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
                "camera": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": False,
                },
            },
        },
    ]
    for robot_cfg in robot_cfgs:
        cfg["robots"] = [robot_cfg]

        # Set the number of envs here!
        num_envs = 3

        configs = []
        for i in range(num_envs):
            # for some reason cfg is giving zero norm quaternion error, so using tiago_primitives.yaml
            config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
            config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
            config["scene"]["load_object_categories"] = ["floors", "coffee_table"]
            config["objects"].append({
                "type": "PrimitiveObject",
                "name": "eef_marker_0",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [1, 0, 0, 1],
            })
            config["objects"].append({
                "type": "PrimitiveObject",
                "name": "eef_marker_1",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [0, 1, 0, 1],
            })
            # In case want different obstacles in different envs
            # if i == 0:
            #     config["scene"]["load_object_categories"] = ["floors", "coffee_table"]
            # if i == 1:
            #     config["scene"]["load_object_categories"] = ["floors", "sofa"]
            # if i == 2:
            #     config["scene"]["load_object_categories"] = ["floors", "walls"]
            configs.append(config)
        vec_env = og.VectorEnvironment(num_envs, configs)
        robots = []
        objs = []
        floor_touching_base_link_prim_paths_list = []
        for env in vec_env.envs:
            robot = env.scene.robots[0]
            robots.append(robot) 
            objs.append(env.scene.object_registry("name", "coffee_table_fqluyq_0"))

            floor_touching_base_link_prim_paths = (
                [os.path.join(robot.prim_path, link_name) for link_name in robot.floor_touching_base_link_names]
                if isinstance(robot, LocomotionRobot)
                else []
            )
            floor_touching_base_link_prim_paths_list.append(floor_touching_base_link_prim_paths)

            robot.reset()

            # Open the gripper(s) to match cuRobo's default state
            for arm_name in robot.gripper_control_idx.keys():
                gripper_control_idx = robot.gripper_control_idx[arm_name]
                robot.set_joint_positions(th.ones_like(gripper_control_idx), indices=gripper_control_idx, normalized=True)

            robot.keep_still()

            for _ in range(5):
                og.sim.step()

            env.scene.update_initial_state()
            env.scene.reset()

        # Create CuRobo instance
        batch_size = 1  # change later to 2 
        n_samples = 5   # change later to 20 

        cmg = CuRoboMotionGenerator(
            robots=robots,
            batch_size=batch_size,
            debug=False,
            use_cuda_graph=False, # change later to True
            collision_activation_distance=0.03,  # Use larger activation distance for better reproducibility
            use_default_embodiment_only=True,
            num_envs=num_envs
        )

        # Sample values for robot
        th.manual_seed(1)
        lo, hi = robot.joint_lower_limits.clone().view(1, -1), robot.joint_upper_limits.clone().view(1, -1)

        if isinstance(robot, HolonomicBaseRobot):
            lo[0, :2] = -0.1
            lo[0, 2:5] = 0.0
            lo[0, 5] = -math.pi
            hi[0, :2] = 0.1
            hi[0, 2:5] = 0.0
            hi[0, 5] = math.pi

        random_qs_list = []
        for i in range(num_envs):
            random_qs = lo + th.rand((n_samples, robot.n_dof)) * (hi - lo)
            random_qs_list.append(random_qs)

        # Test collision with the environment (not including self-collisions)
        collision_results = cmg.check_collisions(qs=random_qs_list)
        
        # target_pos, target_quat = defaultdict(list), defaultdict(list)
        target_pos = [defaultdict(list) for _ in range(num_envs)]
        target_quat = [defaultdict(list) for _ in range(num_envs)]

        # View results
        false_positive = 0
        false_negative = 0

        random_qs_list = th.stack(random_qs_list).permute(1,0,2)
        collision_results = th.stack(collision_results).permute(1,0)
        for i, (q, curobo_has_contact_list) in enumerate(zip(random_qs_list, collision_results)):
            
            for env_idx, robot in enumerate(robots):
                # Set robot to desired qpos
                robot.set_joint_positions(q[env_idx])
                robot.keep_still()
            og.sim.step_physics()

            # To debug
            # cmg.save_visualization(robot.get_joint_positions(), "/scr/chengshu/Downloads/test.obj")

            # Sanity check in the GUI that the robot pose makes sense
            for _ in range(10):
                og.sim.render()

            for env_idx, robot in enumerate(robots):
                floor_prims = [o._prim.GetChildren() for o in vec_env.envs[env_idx].scene.objects if "floor" in o.name]
                floor_prims = np.array(floor_prims).flatten()
                floor_plane_prim_paths = {floor_prim.GetPath().pathString for floor_prim in floor_prims}
                obj = objs[env_idx]

                # Validate that expected collision result is correct
                self_collision_pairs = set()
                floor_contact_pairs = set()
                wheel_contact_pairs = set()
                obj_contact_pairs = set()

                for contact in robot.contact_list():
                    assert contact.body0 in robot.link_prim_paths
                    if contact.body1 in robot.link_prim_paths:
                        self_collision_pairs.add((contact.body0, contact.body1))
                    elif contact.body1 in floor_plane_prim_paths:
                        # breakpoint()
                        if contact.body0 not in floor_touching_base_link_prim_paths_list[env_idx]:
                            floor_contact_pairs.add((contact.body0, contact.body1))
                        else:
                            wheel_contact_pairs.add((contact.body0, contact.body1))
                    elif contact.body1 in obj.link_prim_paths:
                        obj_contact_pairs.add((contact.body0, contact.body1))
                    else:
                        breakpoint()
                        assert False, f"Unexpected contact pair: {contact.body0}, {contact.body1}"

                touching_itself = len(self_collision_pairs) > 0
                touching_floor = len(floor_contact_pairs) > 0
                touching_object = len(obj_contact_pairs) > 0

                curobo_has_contact = curobo_has_contact_list[env_idx].item()
                physx_has_contact = touching_itself or touching_floor or touching_object

                # cuRobo reports contact, but physx reports no contact
                if curobo_has_contact and not physx_has_contact:
                    false_positive += 1
                    print(
                        f"False positive {i}: {curobo_has_contact} vs. {physx_has_contact} (touching_itself/obj/floor: {touching_itself}/{touching_object}/{touching_floor})"
                    )

                # physx reports contact, but cuRobo reports no contact
                elif not curobo_has_contact and physx_has_contact:
                    false_negative += 1
                    print(
                        f"False negative {i}: {curobo_has_contact} vs. {physx_has_contact} (touching_itself/obj/floor: {touching_itself}/{touching_object}/{touching_floor})"
                    )

                # neither cuRobo nor physx reports contact, valid planning goals
                elif not curobo_has_contact and not physx_has_contact:
                    print(f"All good {i}")
                    for arm_name in robot.arm_names:
                        eef_pos, eef_quat = robot.get_eef_pose(arm_name)
                        target_pos[env_idx][robot.eef_link_names[arm_name]].append(eef_pos)
                        target_quat[env_idx][robot.eef_link_names[arm_name]].append(eef_quat)
                # breakpoint()

        print(
            f"Collision checking false positive: {false_positive / n_samples}, false negative: {false_negative / n_samples}."
        )
        assert (
            false_positive / n_samples == 0.0
        ), f"Collision checking false positive rate: {false_positive / n_samples}, should be == 0.0."
        assert (
            false_negative / n_samples == 0.0
        ), f"Collision checking false negative rate: {false_negative / n_samples}, should be == 0.0."

        for env_idx, robot in enumerate(robots):
            vec_env.envs[env_idx].scene.reset()

            for arm_name in robot.arm_names:
                target_pos[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_pos[env_idx][robot.eef_link_names[arm_name]], dim=0)
                target_quat[env_idx][robot.eef_link_names[arm_name]] = th.stack(target_quat[env_idx][robot.eef_link_names[arm_name]], dim=0)

            # Cast defaultdict to dict
            target_pos[env_idx] = dict(target_pos[env_idx])
            target_quat[env_idx] = dict(target_quat[env_idx])

            print(f"Planning for {len(target_pos[env_idx][robot.eef_link_names[robot.default_arm]])} eef targets...")

            # Make sure robot is kept still for better determinism before planning
            robot.keep_still()
            og.sim.step_physics()

        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
        successes, traj_paths, temp_list = cmg.compute_trajectories(
            target_pos_list=target_pos,
            target_quat_list=target_quat,
            is_local=False,
            max_attempts=15,
            timeout=60.0,
            ik_fail_return=5,
            enable_finetune_trajopt=True,
            finetune_attempts=1,
            return_full_result=False,
            success_ratio=1.0,
            attached_obj=None,
        )

        # inefficient way of performing .permute(1,0)
        traj_paths_transposed = []
        for trial_idx in range(len(traj_paths[1])):
            traj_path_all_envs = [traj_paths[0][trial_idx], traj_paths[1][trial_idx], traj_paths[2][trial_idx]]
            traj_paths_transposed.append(traj_path_all_envs)
        traj_paths = traj_paths_transposed

        # # Make sure collision-free trajectories are generated
        # success_rate = successes.double().mean().item()
        # print(f"Collision-free trajectory generation success rate: {success_rate}")
        # assert success_rate == 1.0, f"Collision-free trajectory generation success rate: {success_rate}"

        # 1cm and 3 degrees error tolerance for prismatic and revolute joints, respectively
        error_tol = th.tensor(
            [0.01 if joint.joint_type == "PrismaticJoint" else 3.0 / 180.0 * math.pi for joint in robot.joints.values()]
        )

        # for bypass_physics in [True, False]:
        for bypass_physics in [True]:
            for traj_idx, (success, traj_path_all_envs) in enumerate(zip(successes, traj_paths)):
                
                for env_idx, env in enumerate(vec_env.envs):

                    print(f"============ Motion Planning in env {env_idx}, trial {traj_idx} ==============")
                    breakpoint()
                    robot = robots[env_idx]
                    traj_path = traj_path_all_envs[env_idx]
                    floor_prims = [o._prim.GetChildren() for o in vec_env.envs[env_idx].scene.objects if "floor" in o.name]
                    floor_prims = np.array(floor_prims).flatten()
                    floor_plane_prim_paths = {floor_prim.GetPath().pathString for floor_prim in floor_prims}

                    if not success:
                        continue

                    # Reset the environment
                    env.scene.reset()

                    breakpoint()
                    # Move the markers to the desired eef positions. Convert from robot[i] frame to world frame 
                    robot_pose = robot.get_position_orientation()
                    T_robot_world = th.eye(4)
                    T_robot_world[:3, :3] = T.quat2mat(robot_pose[1])
                    T_robot_world[:3, 3] = robot_pose[0]
                    T_p_robot = th.cat([temp_list[traj_idx], th.tensor([1.0])])
                    T_p_world = T_robot_world @ T_p_robot
                    eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]
                    for marker, arm_name in zip(eef_markers, robot.arm_names):
                        eef_link_name = robot.eef_link_names[arm_name]
                        # marker.set_position_orientation(position=target_pos[-1][eef_link_name][traj_idx])
                        marker.set_position_orientation(position=T_p_world[:3])

                    q_traj = cmg.path_to_joint_trajectory(traj_path)
                    # joint_positions_set_point = []
                    # joint_positions_response = []
                    for i, q in enumerate(q_traj):
                        if bypass_physics:
                            print(f"Teleporting waypoint {i}/{len(q_traj)}")
                            robot.set_joint_positions(q)
                            robot.keep_still()
                            og.sim.step()
                            for contact in robot.contact_list():
                                assert contact.body0 in robot.link_prim_paths
                                if (
                                    contact.body1 in floor_plane_prim_paths
                                    and contact.body0 in floor_touching_base_link_prim_paths_list[env_idx]
                                ):
                                    continue
                                if th.tensor(list(contact.impulse)).norm() == 0:
                                    continue
                                print(f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}")
                                assert (
                                    False
                                ), f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}"
                        else:
                            # Convert target joint positions to action
                            q = q.cpu()
                            action = robot.q_to_action(q)

                            num_repeat = 3
                            for j in range(num_repeat):
                                print(f"Executing waypoint {i}/{len(q_traj)}, step {j}")
                                env.step(action)

                                for contact in robot.contact_list():
                                    assert contact.body0 in robot.link_prim_paths
                                    if (
                                        contact.body1 in floor_plane_prim_paths
                                        and contact.body0 in floor_touching_base_link_prim_paths_list[env_idx]
                                    ):
                                        continue
                                    if th.tensor(list(contact.impulse)).norm() == 0:
                                        continue
                                    print(f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}")
                                    # Controller is not perfect, so collisions might happen
                                    # assert False, f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}"

                                cur_joint_positions = robot.get_joint_positions()
                                if ((cur_joint_positions - q).abs() < error_tol).all():
                                    break

        og.clear()

        del cmg
        gc.collect()
        th.cuda.empty_cache()

test_curobo()