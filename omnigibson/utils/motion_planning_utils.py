import numpy as np
from ompl import base as ob
from ompl import geometric as ompl_geo

import omnigibson as og
from omnigibson.object_states import ContactBodies
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import RigidContactAPI
from pxr import PhysicsSchemaTools, Gf

num_checks = [0]
def plan_base_motion(
    robot,
    end_conf,
    context,
    planning_time = 100.0,
    **kwargs
):
    def state_valid_fn(q):
        x = q.getX()
        y = q.getY()
        yaw = q.getYaw()
        pose = ([x, y, 0.0], T.euler2quat((0, 0, yaw)))
        state_valid = not detect_robot_collision(context, pose)
        num_checks[0] += 1
        return state_valid

    pos = robot.get_position()
    yaw = T.quat2euler(robot.get_orientation())[2]
    start_conf = (pos[0], pos[1], yaw)

    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-3.0)
    bounds.setHigh(3.0)
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.RRTConnect(si)
    ss.setPlanner(planner)

    start = ob.State(space)
    start().setX(start_conf[0])
    start().setY(start_conf[1])
    start().setYaw(T.wrap_angle(start_conf[2]))
    print(start)

    goal = ob.State(space)
    goal().setX(end_conf[0])
    goal().setY(end_conf[1])
    goal().setYaw(T.wrap_angle(end_conf[2]))
    print(goal)

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        sol_path = ss.getSolutionPath()
        print("Num collision checks: ", num_checks[0])
        return_path = []
        for i in range(sol_path.getStateCount()):
            x = sol_path.getState(i).getX()
            y = sol_path.getState(i).getY()
            yaw = sol_path.getState(i).getYaw()
            return_path.append([x, y, yaw])
        return remove_unnecessary_rotations(return_path)
    return None

def plan_arm_motion(
    robot,
    end_conf,
    context,
    planning_time = 100.0,
    **kwargs,
):
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    dim = len(joint_control_idx)

    def state_valid_fn(q):
        joint_pos = [q[i] for i in range(dim)]
        # state_valid = not detect_robot_collision(robot)
        return arm_planning_validity_fn(context, joint_pos)
    
    # create an SE2 state space
    space = ob.RealVectorStateSpace(dim)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(dim)
    joints = np.array([joint for joint in robot.joints.values()])
    arm_joints = joints[joint_control_idx]
    for i, joint in enumerate(arm_joints):
        bounds.setLow(i, float(joint.lower_limit))
        bounds.setHigh(i, float(joint.upper_limit))
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.RRTConnect(si)
    ss.setPlanner(planner)

    start_conf = robot.get_joint_positions()[joint_control_idx]
    start = ob.State(space)
    for i in range(dim):
        start[i] = float(start_conf[i])

    goal = ob.State(space)
    for i in range(dim):
        goal[i] = float(end_conf[i])
    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()

        sol_path = ss.getSolutionPath()
        return_path = []
        for i in range(sol_path.getStateCount()):
            joint_pos = [sol_path.getState(i)[j] for j in range(dim)]
            return_path.append(joint_pos)
        return return_path
    return None

# def detect_robot_collision(robot, filter_objs=[]):
#     filter_categories = ["floors"]
    
#     obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
#     if obj_in_hand is not None:
#         filter_objs.append(obj_in_hand)

#     collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

#     for col_prim in collision_prims:
#         tokens = col_prim.prim_path.split("/")
#         obj_prim_path = "/".join(tokens[:-1])
#         col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
#         if col_obj.category in filter_categories:
#             collision_prims.remove(col_prim)

#     return len(collision_prims) > 0 or detect_self_collision(robot)


# Moves robot and detects robot collisions with the environment, but not with itself
def detect_robot_collision(context, pose):
    translation = pose[0]
    orientation = pose[1]
    # context.robot_prim.set_local_poses(np.array([translation]), np.array([orientation]))
    translation = Gf.Vec3d(*np.array(translation, dtype=float))
    context.robot_prim.GetAttribute("xformOp:translate").Set(translation)

    orientation = np.array(orientation, dtype=float)[[3, 0, 1, 2]]
    context.robot_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation)) 

    for link in context.robot_meshes_copy:
        for mesh in context.robot_meshes_copy[link]:
            mesh_id = PhysicsSchemaTools.encodeSdfPath(mesh.prim_path)
            if mesh._prim.GetTypeName() == "Mesh":
                if og.sim.psqi.overlap_mesh_any(*mesh_id):
                    return True
            else:
                if og.sim.psqi.overlap_shape_any(*mesh_id):
                    return True
    return False

# Detects robot collisions with the environment, but not with itself
def detect_robot_collision_in_sim(robot, filter_objs=[]):
    filter_categories = ["floors"]
    
    obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    if obj_in_hand is not None:
        filter_objs.append(obj_in_hand)

    collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

    for col_prim in collision_prims:
        tokens = col_prim.prim_path.split("/")
        obj_prim_path = "/".join(tokens[:-1])
        col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
        if col_obj.category in filter_categories:
            collision_prims.remove(col_prim)

    return len(collision_prims) > 0
    

# Sets joint positions of robot and detects robot collisions with the environment and itself
def arm_planning_validity_fn(context, joint_pos):
    arm_links = context.robot.manipulation_link_names
    link_poses = context.fk_solver.get_link_poses(joint_pos, arm_links)

    for link in arm_links:
        pose = link_poses[link]
        if link in context.robot_meshes_copy.keys():
            for mesh, relative_pose in zip(context.robot_meshes_copy[link], context.robot_meshes_relative_poses[link]):
                mesh_pose = T.pose_transform(*pose, *relative_pose)
                translation = Gf.Vec3d(*np.array(mesh_pose[0], dtype=float))
                mesh._prim.GetAttribute("xformOp:translate").Set(translation)
                orientation = np.array(mesh_pose[1], dtype=float)[[3, 0, 1, 2]]
                mesh._prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation))

    # Define function for checking overlap
    valid_hit = False
    mesh_hit = None
    links = []

    def overlap_callback(hit):
        nonlocal valid_hit
        nonlocal mesh_hit
        
        valid_hit = hit.rigid_body not in context.collision_pairs_dict[mesh_hit]
        # test = hit.rigid_body not in context.collision_pairs_dict[mesh_hit]
        # if test:
        #     print("hit body")
        #     print(mesh_hit)
        #     print(hit.rigid_body)
        #     link_a = mesh_hit.split("/")[-1]
        #     link_b = hit.rigid_body.split("/")[-1]
        #     pair = {link_a, link_b}
        #     if pair not in links:
        #         links.append({link_a, link_b})
        # return True

        return not valid_hit

    def check_overlap():
        nonlocal valid_hit
        nonlocal mesh_hit
        valid_hit = False

        for link in context.robot_meshes_copy:
            for mesh in context.robot_meshes_copy[link]:
                if valid_hit:
                    return valid_hit
                mesh_id = PhysicsSchemaTools.encodeSdfPath(mesh.prim_path)
                mesh_hit = mesh.prim_path
                if mesh._prim.GetTypeName() == "Mesh":
                    og.sim.psqi.overlap_mesh(*mesh_id, reportFn=overlap_callback)
                else:
                    og.sim.psqi.overlap_shape(*mesh_id, reportFn=overlap_callback)
            
        # print([list(link) for link in links])
        return valid_hit
    
    return not check_overlap()


def detect_self_collision(robot):
    contacts = robot.contact_list()
    robot_links = [link.prim_path for link in robot.links.values()]
    disabled_pairs = [set(p) for p in robot.disabled_collision_pairs]
    for c in contacts:
        link0 = c.body0.split("/")[-1]
        link1 = c.body1.split("/")[-1]
        if {link0, link1} not in disabled_pairs and c.body0 in robot_links and c.body1 in robot_links:
            return True
    return False

def detect_hand_collision(robot, joint_pos, control_idx):
    robot.set_joint_positions(joint_pos, control_idx)
    return detect_robot_collision(robot)

def remove_unnecessary_rotations(path):
    for start_idx in range(len(path) - 1):
        start = np.array(path[start_idx][:2])
        end = np.array(path[start_idx + 1][:2])
        segment = end - start
        theta = np.arctan2(segment[1], segment[0])
        path[start_idx] = (start[0], start[1], theta)
    return path