import pickle
import omnigibson as og
import torch as th
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy

with open("/home/arpit/test_projects/mimicgen/kwargs.pickle", "rb") as f:
    kwargs = pickle.load(f)
    # kwargs["scene"] = {"type": "Scene"}
env = og.Environment(configs=kwargs)

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

# with open("/home/arpit/test_projects/mimicgen/scene_0.pickle", "rb") as f:
with open("/home/arpit/test_projects/mimicgen/debug_no_valid_pose.pickle", "rb") as f:
    scene_0 = pickle.load(f)
og.sim.load_state(scene_0, serialized=False)

for _ in range(20): og.sim.step()

# # 1.
# target_pos = {'base_footprint': th.tensor([[     1.148,      0.588,     -0.000],
#         [     1.148,      0.588,     -0.000],
#         [     1.148,      0.588,     -0.000]])}
# target_quat = {'base_footprint': th.tensor([[    -0.000,      0.000,      0.999,     -0.033],
#         [    -0.000,      0.000,      0.999,     -0.033],
#         [    -0.000,      0.000,      0.999,     -0.033]])}

# 2.
# target_pos = {'base_footprint': th.tensor([[     1.247,      0.477,     -0.000],
#         [     1.247,      0.477,     -0.000],
#         [     1.247,      0.477,     -0.000]])}
# target_quat = {'base_footprint': th.tensor([[    -0.000,      0.000,      0.993,      0.115],
#         [    -0.000,      0.000,      0.993,      0.115],
#         [    -0.000,      0.000,      0.993,      0.115]])}

batch_size = 6
embodiment_selection = "base"

primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=True)

obj =  env.scene.object_registry("name", "teacup")
eef_pose = {'left': (th.tensor([ 0.546, -0.064,  1.046]), th.tensor([ 0.867,  0.370, -0.276, -0.187])), 
            'right': (th.tensor([ 0.505, -0.436,  1.001]), th.tensor([0.634, 0.746, 0.062, 0.196]))}
action_generator = primitive._navigate_to_obj(obj=obj, eef_pose=eef_pose)
next(iter(action_generator))


# motion_kwargs = dict(
#                 trajopt_tsteps=32, # originally 32
#                 collision_checker_type=lazy.curobo.geom.sdf.world.CollisionCheckerType.MESH,
#                 use_cuda_graph=True,
#                 num_ik_seeds=128,
#                 num_batch_ik_seeds=128,
#                 num_batch_trajopt_seeds=1,
#                 num_trajopt_noisy_seeds=1,
#                 ik_opt_iters=100,
#                 optimize_dt=True,
#                 num_trajopt_seeds=4, # originally 4
#                 num_graph_seeds=4, # originally 4
#                 interpolation_dt=0.03,
#                 collision_activation_distance=0.005,
#                 self_collision_check=True,
#                 maximum_trajectory_dt=None,
#                 fixed_iters_trajopt=True,
#                 finetune_trajopt_iters=100,
#                 finetune_dt_scale=1.05,
#                 position_threshold=0.005,
#                 rotation_threshold=0.1, # originally 0.05
#             )
# motion_generator = CuRoboMotionGenerator(
#                 robot=robot,
#                 batch_size=batch_size,
#                 collision_activation_distance=0.005,
#                 motion_cfg_kwargs=motion_kwargs,
#             )
# results, traj_paths = motion_generator.compute_trajectories(
#             target_pos=target_pos,
#             target_quat=target_quat,
#             initial_joint_pos=None,
#             is_local=False,
#             max_attempts=100,
#             timeout=60.0,
#             ik_fail_return=50,
#             enable_finetune_trajopt=True,
#             finetune_attempts=50, # originally 1
#             return_full_result=True,
#             success_ratio=1.0 / batch_size,
#             attached_obj=None,
#             attached_obj_scale=None,
#             motion_constraint=None,
#             skip_obstacle_update=False,
#             ik_only=False,
#             ik_world_collision_check=True,
#             emb_sel=embodiment_selection,
#         )

# while True:
#     successes = results[0].success 
#     print("successes", successes)
#     breakpoint()

#     success_idx = th.where(successes)[0].cpu()
#     if len(success_idx) == 0:
#         traj_path = traj_paths[0]
#     else: 
#         traj_path = traj_paths[success_idx[0]]

#     q_traj = motion_generator.path_to_joint_trajectory(
#         traj_path, get_full_js=True, emb_sel=embodiment_selection
#     ).cpu()

#     q_traj = th.stack(primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))


#     for i, joint_pos in enumerate(q_traj):
#         action = robot.q_to_action(joint_pos)
#         env.step(primitive._postprocess_action(action))

#     breakpoint()
