import os

import yaml
import numpy as np

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

from omnigibson import object_states
from omnigibson.utils.object_state_utils import sample_kinematics
from pxr import PhysxSchema

from omnigibson.controllers.controller_base import ControlType


MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
PREDICATE_SAMPLING_Z_OFFSET = 0.2
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10
    

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)


def _execute_release(robot):
    action = _empty_action(robot)
    controller_name = "gripper_{}".format("0")
    action[robot.controller_action_idx[controller_name]] = 1.0
    robot.release_grasp_immediately()
    # og.sim.step()
    for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
        # Otherwise, keep applying the action!
        yield action

    # Do nothing for a bit so that AG can trigger.
    # for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
    #     yield _empty_action(robot)

def _empty_action(robot):
    action = np.zeros(robot.action_dim)
    for name, controller in robot._controllers.items():
        joint_idx = controller.dof_idx
        action_idx = robot.controller_action_idx[name]
        if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
            action[action_idx] = robot.get_joint_positions()[joint_idx]

    return action

class UndoableContext(object):
    def __init__(self, robot):
        self.robot = robot

    def __enter__(self):
        self.obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm]
        # Store object in hand and the link of the object attached to the robot gripper to manually restore later
        if self.obj_in_hand is not None:
            obj_ag_link_path = self.robot._ag_obj_constraint_params[self.robot.default_arm]['ag_link_prim_path']
            for link in self.obj_in_hand._links.values():
                if link.prim_path == obj_ag_link_path:
                    self.obj_in_hand_link = link
                    break

        self.state = og.sim.dump_state(serialized=False)
        og.sim._physics_context.set_gravity(value=0.0)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
            obj.keep_still()

    def __exit__(self, *args):
        og.sim.load_state(self.state, serialized=False)
        og.sim.step()
        # from IPython import embed; embed()
        if self.obj_in_hand is not None and not self.robot._ag_obj_constraint_params[self.robot.default_arm]:
            self.robot._establish_grasp(ag_data=(self.obj_in_hand, self.obj_in_hand_link))
        og.sim._physics_context.set_gravity(value=-9.81)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(True)
        og.sim.step()

def _sample_pose_with_object_and_predicate(robot, predicate, held_obj, target_obj):
    with UndoableContext(robot):
        robot._release_grasp()
        pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
        result = sample_kinematics(
            pred_map[predicate],
            held_obj,
            target_obj,
            use_ray_casting_method=True,
            max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
            skip_falling=True,
            z_offset=PREDICATE_SAMPLING_Z_OFFSET,
        )

        # self.robot._establish_grasp()
        pos, orn = held_obj.get_position_orientation()
        return pos, orn

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    grasp_obj = DatasetObject(
        name="potato",
        category="potato",
        model="lqjear"
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    start_joint_pos = np.array(
        [
            0.0,
            0.0,  # wheels
            0.2,  # trunk
            0.0,
            1.1707963267948966,
            0.0,  # head
            1.4707963267948965,
            -0.4,
            1.6707963267948966,
            0.0,
            1.5707963267948966,
            0.0,  # arm
            0.05,
            0.05,  # gripper
        ]
    )
    robot.set_position([0.0, -0.5, 0.05])
    robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
    robot.set_joint_positions(start_joint_pos)
    og.sim.step()

    replay_controller(env, "grasp.yaml")
    print(robot.is_grasping())
    print(robot._ag_obj_in_hand)
    test = _sample_pose_with_object_and_predicate(robot, object_states.OnTop, grasp_obj, table)
    execute_controller(_execute_release(robot), env)
    print(robot._ag_obj_in_hand)
    print(robot.is_grasping())
    print(robot._ag_release_counter["0"])
    pause(100)


if __name__ == "__main__":
    main()
