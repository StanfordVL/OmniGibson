import json
import h5py
import omnigibson as og
import torch as th
import random
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
from omnigibson import object_states


seed = 0
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)

# Load the scene from the hdf5 file
f = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_tidy_table.hdf5", "r")
config = f["data"].attrs["config"]
config = json.loads(config)

# Custom changes
config["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
reset_base_pose = (config["robots"][0]["position"], config["robots"][0]["orientation"])
config["robots"][0]["position"] = [0.0, 0.0, 0.0]
config["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]
# config["init_curobo"] = True
# config["env"]["flatten_obs_space"] = True
config["robots"][0]["reset_joint_pos"] =  [
                5.0000,
                1.5000,
                0.000,
                0.000,
                0.000,
                -1.571, # 6 virtual base joint 
                0.5,
                -1.0,
                -0.8,
                -0.0000, # 4 torso joints
                -0.000,
                0.000,
                1.8944,
                1.8945,
                -0.9848,
                -0.9849,
                1.5612,
                1.5621,
                0.9097,
                0.9096,
                -1.5544,
                -1.5545,
                0.0500,
                0.0500,
                0.0500,
                0.0500,
            ]

# Sensor update
config["robots"][0]["obs_modalities"] = ["rgb", "depth_linear", "seg_instance"]
config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = 256
config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = 256
config["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 25.0

env = og.Environment(configs=config)


controller_config = {
    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
    "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
}

env.robots[0].reload_controllers(controller_config=controller_config)
# env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]
robot.set_position_orientation(position=th.tensor(reset_base_pose[0]), orientation=th.tensor(reset_base_pose[1]))
for _ in range(5): og.sim.step()

# set state
import pickle
state = pickle.load(open("/home/arpit/test_projects/OmniGibson/soft_vis_test_state2.pickle", "rb"))
og.sim.load_state(state)
for _ in range(20): og.sim.step()
# breakpoint()

primitive = StarterSemanticActionPrimitives(env,
                                            env.robots[0],
                                            enable_head_tracking=True,
                                            curobo_batch_size=6,
                                            curobo_use_cuda_graph=False,
                                            use_base_pose_hack=False,
                                            real_robot_mode=False,)


target_pos = {'base_link': th.tensor([[ 6.177, -0.698, -0.019],
        [ 6.177, -0.698, -0.019],
        [ 6.177, -0.698, -0.019],
        [ 6.177, -0.698, -0.019],
        [ 6.177, -0.698, -0.019],
        [ 6.177, -0.698, -0.019]]), 'eyes': th.tensor([[ 6.162, -0.318,  1.369],
        [ 6.162, -0.318,  1.369],
        [ 6.162, -0.318,  1.369],
        [ 6.162, -0.318,  1.369],
        [ 6.162, -0.318,  1.369],
        [ 6.162, -0.318,  1.369]])}
target_quat = {'base_link': th.tensor([[-0.002, -0.005,  0.717,  0.697],
        [-0.002, -0.005,  0.717,  0.697],
        [-0.002, -0.005,  0.717,  0.697],
        [-0.002, -0.005,  0.717,  0.697],
        [-0.002, -0.005,  0.717,  0.697],
        [-0.002, -0.005,  0.717,  0.697]]), 'eyes': th.tensor([[0.156, 0.009, 0.019, 0.988],
        [0.156, 0.009, 0.019, 0.988],
        [0.156, 0.009, 0.019, 0.988],
        [0.156, 0.009, 0.019, 0.988],
        [0.156, 0.009, 0.019, 0.988],
        [0.156, 0.009, 0.019, 0.988]])}
eyes_target_pos = th.tensor([ 6.132, -0.205,  1.092])
eyes_target_quat = th.tensor([    -0.000,     -0.000,     -0.653,      0.757])

teacup_601 = env.scene.object_registry("name", "teacup_601")
teacup_601.set_position_orientation(position=eyes_target_pos, orientation=eyes_target_quat)
for _ in range(20): og.sim.step()

# breakpoint()
emb_sel = "base"
mp_results, traj_paths = primitive._motion_generator.compute_trajectories(
    target_pos=target_pos,
    target_quat=target_quat,
    is_local=False,
    max_attempts=50,
    timeout=60.0,
    ik_fail_return=50,
    enable_finetune_trajopt=True,
    finetune_attempts=1,
    return_full_result=True,
    success_ratio=1.0,
    emb_sel=emb_sel,
    attached_obj=None,
    attached_obj_scale=None,
    eyes_target_pos=eyes_target_pos,
    eyes_target_quat=eyes_target_quat,
)
successes = mp_results[0].success 
print("successes: ", successes)
success_status, traj_path = successes[0], traj_paths[0]
# breakpoint()

q_traj = primitive._motion_generator.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
q_traj = q_traj.cpu()
# print("q_traj shape ", q_traj.shape)
q_traj = th.stack(primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
# print("q_traj shape after interpolation ", q_traj.shape)

init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
mp_actions = []
for j_pos in q_traj:
    action = robot.q_to_action(j_pos).cpu().numpy()
    action[robot.gripper_action_idx["right"]] = -1.0
    action[robot.arm_action_idx["left"]] = init_left_arm_pos
    env.step(action)
breakpoint()


    

