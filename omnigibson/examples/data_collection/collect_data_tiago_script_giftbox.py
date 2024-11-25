import os
import pdb

import yaml
import torch
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import BimanualKeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T

from collect_data_tiago_script_cup import generate_action, generate_waypoint_sequence, move_to_waypoints, _empty_action, send_to_target_pose, format_action, get_eef_pos_orn, close_gripper, open_gripper, move_to_waypoints

from omnigibson.utils.control_utils import orientation_error

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False


def main():
    # config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    # Rs_int_task_set_up_a_coffee_station_in_your_kitchen_0_0_template.json
    config_filename = os.path.join(og.example_config_path, "tiago_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    activity_name = "test_cabinet"
    # activity_name = "test_tiago_cup"
    activity_name = "set_up_a_coffee_station_in_your_kitchen"
    activity_name = "putting_dirty_dishes_in_sink"
    activity_name = "make_microwave_popcorn"
    activity_name = "test_tiago_plate"
    activity_name = "test_tiago_giftbox"

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
    pdb.set_trace()

    state = og.sim.dump_state()
    og.sim.stop()
    gift_box = env.scene.object_registry("name", "gift_box")
    gift_box.links['base_link'].density = 10
    # coffee_cup.links['base_link'].friction = 0.01 # friction is not in the link object
    # giftbox = env.scene.object_registry("name", "gift_box")
    # giftbox.links['base_link'].density = 100
    og.sim.play()
    og.sim.load_state(state)
    for _ in range(10): og.sim.step()


    # Create teleop controller
    action_generator = BimanualKeyboardRobotController(robot=robot)

    def start_teleop(env=env, robot=robot, action_generator=action_generator):
        for _ in range(200):
            action = action_generator.get_teleop_action_bimanual()
            next_obs, reward, terminated, truncated, info = env.step(action=action)
        print('arm_left:', get_eef_pos_orn(robot, 'left'))
        print('arm_right:', get_eef_pos_orn(robot, 'right'))
    
    def render(steps):
        for _ in range(steps): og.sim.render()

    pdb.set_trace()

    # pre_grasp_pos_left = paper_cup.get_position_orientation()[0] - torch.Tensor([0.1, 0, 0])
    # pre_grasp_ori_left = paper_cup.get_position_orientation()[1]
    # # concat the position and orientation
    # pre_grasp_left = torch.cat([pre_grasp_pos_left, pre_grasp_ori_left])

    # pre_grasp_pos_right = paper_cup.get_position_orientation()[0] - torch.Tensor([0.1, 0, 0])
    # pre_grasp_ori_right = paper_cup.get_position_orientation()[1]
    # # concat the position and orientation
    # pre_grasp_right = torch.cat([pre_grasp_pos_right, pre_grasp_ori_right])

    # send_to_target_pose(robot, 'arm_left', torch.Tensor([0.6, 0.2228, 0.7436]), torch.Tensor([-0.6877,  0.6333,  0.2396,  0.2618]))

    waypoints_list = [
    {
        "arm_left":   
        (      
            (torch.Tensor([0.4684,  0.17,  0.7864]), torch.Tensor([-0.6310,  0.5976,  0.3482,  0.3514]), -1),
            (torch.Tensor([0.4684,  0.08,  0.7864]), torch.Tensor([-0.6310,  0.5976,  0.3482,  0.3514]), 0),
        ),
        "arm_right":
        (   
            (torch.Tensor([ 0.4779, -0.17,  0.7829 ]), torch.Tensor([-0.6344, -0.5998,  0.3544, -0.3350]), -1),
            (torch.Tensor([ 0.4779, -0.08,  0.7829 ]), torch.Tensor([-0.6344, -0.5998,  0.3544, -0.3350]), 0), 
        ),
    },
    {
        "arm_left":
        (
            (torch.Tensor([0.65,  0.08,  0.7864]), torch.Tensor([-0.6310,  0.5976,  0.3482,  0.3514]), 0),
            (torch.Tensor([0.5,  0.2,  0.7864]), torch.Tensor([-0.6310,  0.5976,  0.3482,  0.3514]), 0),

            (None, None, 0),
        ),
        "arm_right":
        (
            (torch.Tensor([ 0.65, -0.08,  0.7829 ]), torch.Tensor([-0.6344, -0.5998,  0.3544, -0.3350]), 0), #
            (torch.Tensor([ 0.5, -0.2,  0.7829 ]), torch.Tensor([-0.6344, -0.5998,  0.3544, -0.3350]), 0), # 
            (None, None, 0),
        )
    }
    ]


    for waypoint in waypoints_list:
        move_to_waypoints(waypoint, env, robot)

    pdb.set_trace()
    print('now the waypoint is reached')

    print("Data saved")
    env.save_data()

    og.shutdown()


if __name__ == "__main__":
    main()
