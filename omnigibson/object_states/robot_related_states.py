import collections
import numpy as np
import omnigibson as og
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin, RelativeObjectState

from omni.syntheticdata import helpers
from omnigibson.robots import tiago


_IN_REACH_DISTANCE_THRESHOLD = 2.0

_IN_FOV_PIXEL_FRACTION_THRESHOLD = 0.05


class RobotStateMixin:
    @property
    def robot(self):
        from omnigibson.robots.robot_base import BaseRobot
        assert isinstance(self.obj, BaseRobot), "This state only works with robots."
        return self.obj


class IsGrasping(RelativeObjectState, BooleanStateMixin, RobotStateMixin):
    def _get_value(self, obj):
        # TODO: Make this work with non-assisted grasping
        return any(
            self.robot._ag_obj_in_hand[arm] == obj 
            for arm in self.robot.arm_names
        )


# class InReachOfRobot(AbsoluteObjectState, BooleanStateMixin):
#     def _compute_value(self):
#         robot = _get_robot(self.simulator)
#         if not robot:
#             return False

#         robot_pos = robot.get_position()
#         object_pos = self.obj.get_position()
#         return np.linalg.norm(object_pos - np.array(robot_pos)) < _IN_REACH_DISTANCE_THRESHOLD


class InFOVOfRobot(RelativeObjectState, BooleanStateMixin, RobotStateMixin):
    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [ObjectsInFOVOfRobot]

    def _get_value(self, obj):
        return obj in self.robot.states[ObjectsInFOVOfRobot].get_value()
    
    def _set_value(self, obj, new_value):
        assert new_value == True, "Cannot set InFOVOfRobot to False. Feel free to sample random camera pose."
        assert isinstance(self.robot, tiago), "Setting InFOVOfRobot is currently only supported for Tiago"

        # Get the object's position.
        obj_pos = obj.get_position()

        # Get where the camera needs to point.
        camera_joint_targets = self.robot.get_head_joint_positions_for_target(obj_pos)
        if camera_joint_targets is None:
            return False

        # Set the camera joint positions.
        camera_joint_indices = self.robot.camera_control_idx
        assert len(camera_joint_indices) == len(camera_joint_targets), "Camera joint indices and targets must be same length"
        self.robot.set_joint_positions(camera_joint_targets, indices=camera_joint_indices)

        # TODO: Verify that the object is in FOV
        return True


class ObjectsFractionOfFOVOfRobot(AbsoluteObjectState, RobotStateMixin):
    def _get_value(self):
        # Get the observation
        cam = self.robot.sensors["robot0:eyes_Camera_sensor"]
        insts = cam.get_obs()["seg_instance"].flatten()
        inst_counts = collections.Counter(insts)
        total_pixels = len(insts)

        # Get the key for the instance-to-primitive mapping.
        im = helpers.get_instance_mappings()
        ins_to_prim = {entry[0]: entry[1] for entry in im}

        # Get the counts for each object.
        prim_counts = {ins_to_prim[k]: v for k, v in inst_counts.items() if k in ins_to_prim}
        obj_counts = {og.sim.scene.object_registry("prim_path", k): v for k, v in prim_counts.items()}
        obj_counts[None] = total_pixels - sum(obj_counts.values())
        obj_counts = {k: v / total_pixels for k, v in obj_counts.items()}

        return obj_counts


class ObjectsInFOVOfRobot(AbsoluteObjectState, RobotStateMixin):
    def _get_value(self):
        # Get the observation
        cam = self.robot.sensors["robot0:eyes_Camera_sensor"]
        insts = np.unique(cam.get_obs()["seg_instance"].flatten())

        # Get the key for the instance-to-primitive mapping.
        im = helpers.get_instance_mappings()
        ins_to_prim = {entry[0]: entry[1] for entry in im}

        # Get the relevant objects.
        objs = {og.sim.scene.object_registry("prim_path", ins_to_prim[inst]) for inst in insts if inst in ins_to_prim}
        return objs