import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.transform_utils import mat2pose, pose2mat, relative_pose_transform
from omnigibson.utils.usd_utils import PoseAPI


def setup_environment(flatcache):
    """
    Sets up the environment with or without flatcache based on the flatcache parameter.
    """
    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = flatcache  # Set based on function parameter
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb", "seg_semantic", "seg_instance"],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
            }
        ],
    }

    env = og.Environment(configs=config)
    return env


def camera_pose_test(flatcache):
    env = setup_environment(flatcache)
    robot = env.robots[0]
    env.reset()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]

    # Get vision sensor world pose via directly calling get_position_orientation
    robot_world_pos, robot_world_ori = robot.get_position_orientation()
    sensor_world_pos, sensor_world_ori = vision_sensor.get_position_orientation()

    robot_to_sensor_mat = pose2mat(
        relative_pose_transform(sensor_world_pos, sensor_world_ori, robot_world_pos, robot_world_ori)
    )

    sensor_world_pos_gt = np.array([150.16513062, 150.0, 101.38952637])
    sensor_world_ori_gt = np.array([-0.29444984, 0.29444979, 0.64288365, -0.64288352])

    assert np.allclose(sensor_world_pos, sensor_world_pos_gt, atol=1e-3)
    assert np.allclose(sensor_world_ori, sensor_world_ori_gt, atol=1e-3)

    # Now, we want to move the robot and check if the sensor pose has been updated
    old_camera_local_pose = vision_sensor.get_position_orientation("parent")

    robot.set_position_orientation(position=[100, 100, 100])
    new_camera_local_pose = vision_sensor.get_position_orientation("parent")
    new_camera_world_pose = vision_sensor.get_position_orientation()
    robot_pose_mat = pose2mat(robot.get_position_orientation())
    expected_camera_world_pos, expected_camera_world_ori = mat2pose(robot_pose_mat @ robot_to_sensor_mat)
    assert np.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert np.allclose(new_camera_world_pose[0], expected_camera_world_pos, atol=1e-3)
    assert np.allclose(new_camera_world_pose[1], expected_camera_world_ori, atol=1e-3)

    # Then, we want to move the local pose of the camera and check
    # 1) if the world pose is updated 2) if the robot stays in the same position
    old_camera_local_pose = vision_sensor.get_position_orientation("parent")
    vision_sensor.set_position_orientation(position=[10, 10, 10], orientation=[0, 0, 0, 1], frame="parent")
    new_camera_world_pose = vision_sensor.get_position_orientation()
    camera_parent_prim = lazy.omni.isaac.core.utils.prims.get_prim_parent(vision_sensor.prim)
    camera_parent_path = str(camera_parent_prim.GetPath())
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    expected_new_camera_world_pos, expected_new_camera_world_ori = mat2pose(
        camera_parent_world_transform @ pose2mat([[10, 10, 10], [0, 0, 0, 1]])
    )
    assert np.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)
    assert np.allclose(new_camera_world_pose[1], expected_new_camera_world_ori, atol=1e-3)
    assert np.allclose(robot.get_position_orientation()[0], [100, 100, 100], atol=1e-3)

    # Finally, we want to move the world pose of the camera and check
    # 1) if the local pose is updated 2) if the robot stays in the same position
    robot.set_position_orientation(position=[150, 150, 100])
    old_camera_local_pose = vision_sensor.get_position_orientation("parent")
    vision_sensor.set_position_orientation([150, 150, 101.36912537], [-0.29444987, 0.29444981, 0.64288363, -0.64288352])
    new_camera_local_pose = vision_sensor.get_position_orientation("parent")
    assert not np.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert not np.allclose(old_camera_local_pose[1], new_camera_local_pose[1], atol=1e-3)
    assert np.allclose(robot.get_position_orientation()[0], [150, 150, 100], atol=1e-3)

    # Another test we want to try is setting the camera's parent scale and check if the world pose is updated
    camera_parent_prim.GetAttribute("xformOp:scale").Set(lazy.pxr.Gf.Vec3d([2.0, 2.0, 2.0]))
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    camera_local_pose = vision_sensor.get_position_orientation("parent")
    expected_new_camera_world_pos, _ = mat2pose(camera_parent_world_transform @ pose2mat(camera_local_pose))
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert np.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)

    og.clear()


def test_camera_pose_flatcache_on():
    camera_pose_test(True)
