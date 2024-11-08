import pdb
import traceback

import torch as th

import omnigibson as og
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator


def plan_trajectory(cmg, target_pos, target_quat, emb_sel=CuroboEmbodimentSelection.DEFAULT, return_full_result=False):
    return cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=1,
                                    timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1,
                                    return_full_result=return_full_result, success_ratio=1.0, attached_obj=None,
                                    emb_sel=emb_sel, )


def test_curobo():
    # Create env
    cfg = {"env": {"action_frequency": 30, "physics_frequency": 300, }, "scene": {"type": "Scene", }, "objects": [
        {"type": "PrimitiveObject", "name": "eef_marker_0", "primitive_type": "Sphere", "radius": 0.05,
         "visual_only": True, "position": [0, 0, 0], "orientation": [0, 0, 0, 1], "rgba": [1, 0, 0, 1], },
        {"type": "PrimitiveObject", "name": "eef_marker_1", "primitive_type": "Sphere", "radius": 0.05,
         "visual_only": True, "position": [0, 0, 0], "orientation": [0, 0, 0, 1], "rgba": [0, 1, 0, 1], }, ],
           "robots": [{"type": "Tiago", "obs_modalities": "rgb", "position": [0, 0, 0], "orientation": [0, 0, 0, 1],
                       "self_collisions": True, "action_normalize": False, "rigid_trunk": False, "controller_config": {
                   "base": {"name": "JointController", "motor_type": "position", "command_input_limits": None,
                            "use_delta_commands": False, "use_impedances": True},
                   "arm_left": {"name": "JointController", "motor_type": "position", "command_input_limits": None,
                                "use_delta_commands": False, "use_impedances": True, },
                   "arm_right": {"name": "JointController", "motor_type": "position", "command_input_limits": None,
                                 "use_delta_commands": False, "use_impedances": True, },
                   "gripper_left": {"name": "JointController", "motor_type": "position",
                                    "command_input_limits": [-1, 1], "use_delta_commands": False,
                                    "use_impedances": True, },
                   "gripper_right": {"name": "JointController", "motor_type": "position",
                                     "command_input_limits": [-1, 1], "use_delta_commands": False,
                                     "use_impedances": True, }, }, }],

           }

    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]

    # Stablize the robot and update the initial state
    robot.reset()

    # Open the gripper(s) to match cuRobo's default state
    for arm_name in robot.gripper_control_idx.keys():
        grpiper_control_idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_state()
    env.scene.reset()

    # Create CuRobo instance
    cmg = CuRoboMotionGenerator(robot=robot, batch_size=1, use_cuda_graph=True, )
    for _ in range(2):
        for emb_sel in CuroboEmbodimentSelection:
            if emb_sel != CuroboEmbodimentSelection.DEFAULT:
                continue
            try:
                print("Embodiment Selection: ", emb_sel)
                target_pos, target_quat = dict(), dict()
                target_links = []
                if emb_sel == CuroboEmbodimentSelection.BASE:
                    pos, quat = robot.links['base_footprint'].get_position_orientation()
                    target_pos['base_footprint'] = pos + th.tensor([2.0, -2.0, 0.1])
                    target_quat['base_footprint'] = quat
                    target_links.append('base_footprint')
                else:
                    for arm in robot.arm_names:
                        # if arm == "right":
                        #     continue
                        pos, quat = robot.get_eef_pose(arm=arm)
                        target_pos[robot.eef_link_names[arm]] = pos + th.tensor([1.0, 1.0, 0.1])
                        target_quat[robot.eef_link_names[arm]] = quat
                        target_links.append(robot.eef_link_names[arm])

                successes, traj_paths = plan_trajectory(cmg, target_pos, target_quat, emb_sel)
                traj_path = traj_paths[0]
                print("success, traj_path: ", successes, traj_path.position.shape)

                for marker in eef_markers:
                    marker.set_position_orientation(position=th.tensor([0, 0, 0]))
                for target_link, marker in zip(target_links, eef_markers):
                    marker.set_position_orientation(position=target_pos[target_link])

                assert successes
                q_traj = cmg.path_to_joint_trajectory(traj_path, emb_sel)
                effective_joint_indices = cmg.get_effective_joint_names(traj_path, emb_sel)
                for q in q_traj:
                    robot.set_joint_positions(q, indices=effective_joint_indices)
                    robot.keep_still()
                    og.sim.step()
            except:
                traceback.print_exc()
                pdb.set_trace()

    og.shutdown()


if __name__ == "__main__":
    test_curobo()
