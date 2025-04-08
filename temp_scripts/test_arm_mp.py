import os
import pickle
import imageio
import yaml
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy


seed = 0
np.random.seed(seed)
th.manual_seed(seed)

def reset():
    # obs, info = env.reset()
    orn = R.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    robot.set_position_orientation(position=th.tensor([-1.0, 0.0, 0.0]), orientation=orn)
    for _ in range(20): og.sim.step()


robot = "R1"

config_filename = os.path.join(og.example_config_path, "r1_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"]["type"] = "Scene"

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
env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]
primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=False, curobo_batch_size=1)

og.sim.viewer_camera.set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))

for _ in range(10): og.sim.step()

reset()

curr_right_eef_pose = robot.get_eef_pose("right")
# target_pos = {"right_eef_link": curr_right_eef_pose[0] + th.tensor([0.2, 0.0, 0.3])}
# target_quat = {"right_eef_link": curr_right_eef_pose[1]}

# curr_left_eef_pose = robot.get_eef_pose("left")
# target_pos = {"left_eef_link": curr_left_eef_pose[0] + th.tensor([0.2, 0.0, 0.3])}
# target_quat = {"left_eef_link": curr_left_eef_pose[1]}

emb_sel = "arm"

all_rollout_fns = [
    fn
    for fn in primitive._motion_generator.mg[emb_sel].get_all_rollout_instances()
    if isinstance(fn, lazy.curobo.rollout.arm_reacher.ArmReacher)
]
# breakpoint()
# rollout_fn._link_pose_costs[k].weight
init_state = og.sim.dump_state()
for _ in range(10):

    x = np.random.uniform(0.2, 0.4)
    z = np.random.uniform(0.2, 0.5)
    target_pos = {"right_eef_link": curr_right_eef_pose[0] + th.tensor([x, 0.0, z])}
    target_quat = {"right_eef_link": curr_right_eef_pose[1]}

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
    )

    successes = mp_results[0].success 
    print("successes: ", successes)
    success_status, traj_path = successes[0], traj_paths[0]

    q_traj = primitive._motion_generator.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
    q_traj = q_traj.cpu()
    print("q_traj shape ", q_traj.shape)
    # q_traj = th.stack(primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
    # print("q_traj shape after interpolation ", q_traj.shape)
    mp_actions = []
    for j_pos in q_traj:
        action = robot.q_to_action(j_pos).cpu().numpy()
        env.step(action)

    print("initial right eef pos: ", curr_right_eef_pose[0])
    print("target right eef pos: ", target_pos["right_eef_link"])
    print("final right eef pos: ", robot.get_eef_pose("right")[0])
    print("norm: ", th.norm(target_pos["right_eef_link"] - robot.get_eef_pose("right")[0]))

    for _ in range(100): og.sim.step()
    og.sim.load_state(init_state)
    for _ in range(10): og.sim.step()




# print("-------------------")
# print("initial left eef pose: ", curr_left_eef_pose)
# print("final left eef pose: ", robot.get_eef_pose("left"))

k = "right_eef_link"
temp = [rollout_fn._link_pose_costs[k].weight for rollout_fn in all_rollout_fns]
print(temp)
breakpoint()