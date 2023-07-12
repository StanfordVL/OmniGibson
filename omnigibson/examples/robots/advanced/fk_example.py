import argparse
import time

import numpy as np

import omnigibson as og
from omnigibson.objects import PrimitiveObject
from omnigibson.robots import Fetch
from omnigibson.scenes import Scene
from omnigibson.utils.control_utils import FKSolver, IKSolver
import omnigibson.utils.transform_utils as T

import carb
import omni


def main():
    """
    Minimal example of usage of forward kinematics (FK) solver

    This example showcases how to use FK functionality using omniverse's native lula library.
    We iterate through desired positions for the gripper, use IK solver to get the joint positions
    that will reach those positions, and verify FK functionality by checking that the predicted gripper
    positions are close to the initial desired positions.
    """
    scene = Scene()
    og.sim.import_scene(scene)

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([4.32248, -5.74338, 6.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    # Create Fetch robot
    # Note that since we only care about the control of arm joints (including the trunk), we fix the base 
    robot = Fetch(
        prim_path="/World/robot",
        name="robot",
        fixed_base=True,
        controller_config={
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
            }
        }
    )
    og.sim.import_object(robot)

    # Set robot base at the origin
    robot.set_position_orientation(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
    # to be fully initialized after it is imported into the simulator
    og.sim.play()
    og.sim.step()
    # Make sure none of the joints are moving
    robot.keep_still()

    # Create the IK solver 
    control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        default_joint_pos=robot.get_joint_positions()[control_idx],
        eef_name=robot.eef_link_names[robot.default_arm],
    )

    # Create the FK solver 
    fk_solver = FKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
    )

    # Sanity check FK using IK
    query_positions = [[1, 0, 0.8], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
    for query_pos in query_positions:
        # find the joint position for a target eef position using IK
        joint_pos = ik_solver.solve(
            target_pos=query_pos,
            target_quat=None,
            max_iterations=100,
        )
        if joint_pos is None:
            og.log.info("EE position not reachable.")
        else:
            og.log.info("Solution found. Verifying FK with the solution.")

            # find the eef position with FK with the IK's solution joint position
            link_poses = fk_solver.get_link_poses(joint_pos, ["gripper_link"])
            gripper_pose = link_poses["gripper_link"]
            fk_joint_pos, _ = T.pose_transform(*robot.get_position_orientation(), *gripper_pose)

            # verify that the FK's predicted eef position is close to the IK's target eef position
            err = np.linalg.norm(fk_joint_pos - query_pos)
            if err < 0.001:
                og.log.info("Predicted position and the target position are close.")
            else:
                og.log.info("Predicted position and the target position are different.")

        time.sleep(5)


if __name__ == "__main__":
    main()
