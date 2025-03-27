import pickle
import omnigibson as og
import torch as th
import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject


seed = 1
np.random.seed(seed)
th.manual_seed(seed)


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
robot.set_position_orientation(position=th.tensor([-1.0, 0.0, 0.0]))
primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=True)

og.sim.viewer_camera.set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))
env._external_sensors["external_sensor0"].set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))

left_eef_marker = PrimitiveObject(
    relative_prim_path="/left_eef_marker",
    primitive_type="Cube",
    name="left_eef_marker",
    size=th.tensor([0.03, 0.03, 0.03]),
    visual_only=True,
    rgba=th.tensor([1, 0, 0, 1]))
success_base_marker = PrimitiveObject(
    relative_prim_path="/success_base_marker",
    primitive_type="Cube",
    name="success_base_marker",
    size=th.tensor([0.03, 0.03, 0.03]),
    visual_only=True,
    rgba=th.tensor([0, 1, 0, 1]))
failure_base_marker = PrimitiveObject(
    relative_prim_path="/failure_base_marker",
    primitive_type="Cube",
    name="failure_base_marker",
    size=th.tensor([0.03, 0.03, 0.03]),
    visual_only=True,
    rgba=th.tensor([1, 0, 0, 1]))

markers = {
    "left_eef_marker": left_eef_marker, 
    "success_base_marker": success_base_marker, 
    "failure_base_marker": failure_base_marker,
}
og.sim.batch_add_objects([left_eef_marker, success_base_marker, failure_base_marker], [env.scene] * 3)

# robot.set_joint_positions(robot.tucked_default_joint_pos)

sampled_base_poses = {"failure": list(), "success": list()}

for _ in range(10): og.sim.step()
teacup = env.scene.object_registry("name", "teacup")
breakfast_table = env.scene.object_registry("name", "breakfast_table")

# breakpoint()

num_trials = 10
num_base_mp_failures, num_base_sampling_failures = 0, 0
for i in range(num_trials):
    print(f"========= Trial {i} =========")
    primitive.valid_env = True
    primitive.err = "None"
    # sample obj poses
    teacup.states[object_states.OnTop].set_value(other=breakfast_table, new_value=True)
    for _ in range(20): og.sim.step()

    action_generator = primitive._navigate_to_obj(obj=teacup, markers=markers, sampled_base_poses=sampled_base_poses)
    next(iter(action_generator))
    if primitive.err == "BaseMPFailed":
        num_base_mp_failures += 1
    elif primitive.err == "NoValidPose":
        num_base_sampling_failures += 1
    for _ in range(100): og.sim.step()

print("num_base_sampling_failures", num_base_sampling_failures)
print("num_base_mp_failures", num_base_mp_failures)

# ================== Visualization ==================
base_marker_list = []
failures = sampled_base_poses["failure"]
for i in range(len(failures)):
    base_marker = PrimitiveObject(
        relative_prim_path=f"/base_marker_failure_{i}",
        primitive_type="Cube",
        name=f"base_marker_failure_{i}",
        size=th.tensor([0.03, 0.03, 0.03]),
        visual_only=True,
        rgba=th.tensor([1, 0, 0, 1])
    )
    base_marker_list.append(base_marker)
og.sim.batch_add_objects(base_marker_list, [env.scene] * len(base_marker_list))
for i in range(len(failures)):
    base_pos = failures[i]
    base_marker_list[i].set_position_orientation(position=base_pos)

base_marker_list = []
success = sampled_base_poses["success"]
for i in range(len(success)):
    base_marker = PrimitiveObject(
        relative_prim_path=f"/base_marker_success_{i}",
        primitive_type="Cube",
        name=f"base_marker_success_{i}",
        size=th.tensor([0.03, 0.03, 0.03]),
        visual_only=True,
        rgba=th.tensor([0, 1, 0, 1])
    )
    base_marker_list.append(base_marker)
og.sim.batch_add_objects(base_marker_list, [env.scene] * len(base_marker_list))
for i in range(len(success)):
    base_pos = success[i]
    base_marker_list[i].set_position_orientation(position=base_pos)

for _ in range(300): og.sim.step()
breakpoint()

# # ================== Visualization ==================





# # with open("/home/arpit/test_projects/mimicgen/scene_0.pickle", "rb") as f:
# with open("/home/arpit/test_projects/mimicgen/debug_no_valid_pose.pickle", "rb") as f:
#     scene_0 = pickle.load(f)
# og.sim.load_state(scene_0, serialized=False)

# for _ in range(20): og.sim.step()

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

# batch_size = 6
# embodiment_selection = "base"

# primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=True)

# obj =  env.scene.object_registry("name", "teacup")
# eef_pose = {'left': (th.tensor([ 0.546, -0.064,  1.046]), th.tensor([ 0.867,  0.370, -0.276, -0.187])), 
#             'right': (th.tensor([ 0.505, -0.436,  1.001]), th.tensor([0.634, 0.746, 0.062, 0.196]))}
# action_generator = primitive._navigate_to_obj(obj=obj, eef_pose=eef_pose)
# next(iter(action_generator))


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
