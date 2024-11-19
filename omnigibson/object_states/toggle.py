import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.object_states.update_state_mixin import GlobalUpdateStateMixin, UpdateStateMixin
from omnigibson.utils.constants import PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI, absolute_prim_path_to_scene_relative, create_primitive_mesh

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.TOGGLE_LINK_PREFIX = "togglebutton"
m.DEFAULT_SCALE = 0.1
m.CAN_TOGGLE_STEPS = 5


class ToggledOn(AbsoluteObjectState, BooleanStateMixin, LinkBasedStateMixin, UpdateStateMixin, GlobalUpdateStateMixin):

    # List of set of prim paths defining robot finger links belonging to any manipulation robots per scene
    _robot_finger_paths = None

    # Set of objects that are contacting any manipulation robots
    _finger_contact_objs = None

    def __init__(self, obj, scale=None):
        self.scale = scale
        self.value = False
        self.robot_can_toggle_steps = 0
        self.visual_marker = None

        super().__init__(obj)

    @classmethod
    def global_update(cls):
        # Avoid circular imports
        from omnigibson.robots.manipulation_robot import ManipulationRobot

        # Clear finger contact objects since it will be refreshed now
        cls._finger_contact_objs = set()

        # detect marker and hand interaction
        cls._robot_finger_paths = [
            {
                link.prim_path
                for robot in scene.robots
                if isinstance(robot, ManipulationRobot)
                for finger_links in robot.finger_links.values()
                for link in finger_links
            }
            for scene in og.sim.scenes
        ]

        # If there aren't any valid robot link paths, immediately return
        if not any(len(robot_finger_paths) > 0 for robot_finger_paths in cls._robot_finger_paths):
            return

        for scene_idx, (scene, scene_robot_finger_paths) in enumerate(zip(og.sim.scenes, cls._robot_finger_paths)):
            if len(scene_robot_finger_paths) == 0:
                continue
            finger_idxs = [RigidContactAPI.get_body_col_idx(prim_path)[1] for prim_path in scene_robot_finger_paths]
            finger_impulses = RigidContactAPI.get_all_impulses(scene_idx)[:, finger_idxs, :]
            n_bodies = len(finger_impulses)
            touching_bodies = th.any(finger_impulses.reshape(n_bodies, -1), dim=-1)
            touching_bodies_idxs = th.where(touching_bodies)[0]
            if len(touching_bodies_idxs) > 0:
                for idx in touching_bodies_idxs:
                    body_prim_path = RigidContactAPI.get_row_idx_prim_path(scene_idx, idx=idx)
                    obj = scene.object_registry("prim_path", "/".join(body_prim_path.split("/")[:-1]))
                    if obj is not None:
                        cls._finger_contact_objs.add(obj)

    @classproperty
    def metalink_prefix(cls):
        return m.TOGGLE_LINK_PREFIX

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value

        # Choose which color to apply to the toggle marker
        self.visual_marker.color = th.tensor([0, 1.0, 0]) if self.value else th.tensor([1.0, 0, 0])

        return True

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Make sure this object is not cloth
        assert self.obj.prim_type != PrimType.CLOTH, f"Cannot create ToggledOn state for cloth object {self.obj.name}!"

        mesh_name = "mesh_0"
        mesh_prim_path = f"{self.link.prim_path}/{mesh_name}"
        pre_existing_mesh = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mesh_prim_path)
        # Create a primitive mesh if it doesn't already exist
        if not pre_existing_mesh:
            self.scale = m.DEFAULT_SCALE if self.scale is None else self.scale
            # Note: We have to create a mesh (instead of a sphere shape) because physx complains about non-uniform
            # scaling for non-meshes
            mesh = create_primitive_mesh(
                prim_path=mesh_prim_path,
                primitive_type="Sphere",
                extents=1.0,
            )
        else:
            # Infer radius from mesh if not specified as an input
            lazy.omni.isaac.core.utils.bounds.recompute_extents(prim=pre_existing_mesh)
            self.scale = vtarray_to_torch(pre_existing_mesh.GetAttribute("xformOp:scale").Get())

        # Make sure the object updates its meshes, and assert that there's only a single visual mesh
        self.link.update_meshes(trigger_mesh_paths=[mesh_prim_path])
        assert len(self.link.visual_meshes) == 1, "Toggle button must have exactly one visual mesh"
        self.visual_marker = self.link.visual_meshes[mesh_name]
        self.visual_marker.scale = self.scale
        self.visual_marker.initialize()
        self.visual_marker.visible = True
        # Make sure the toggle button is visible
        self.visual_marker.purpose = "default"

    def _update(self):
        # If we're not nearby any fingers, we automatically can't toggle
        if self.obj not in self._finger_contact_objs:
            robot_can_toggle = False
        else:
            # Check to make sure fingers are actually overlapping the toggle button mesh
            trigger_colliders = self.visual_marker.get_colliding_prim_paths()
            all_finger_paths = {path for path_set in self._robot_finger_paths for path in path_set}
            robot_can_toggle = bool(trigger_colliders & all_finger_paths)

        previous_step = self.robot_can_toggle_steps
        if robot_can_toggle:
            self.robot_can_toggle_steps += 1
        else:
            self.robot_can_toggle_steps = 0

        # If step size is different, add this object to the current state update set in its scene
        if previous_step != self.robot_can_toggle_steps:
            self.obj.state_updated()

        if self.robot_can_toggle_steps == m.CAN_TOGGLE_STEPS:
            self.set_value(not self.value)

    @staticmethod
    def get_texture_change_params():
        # By default, it keeps the original albedo unchanged.
        albedo_add = 0.0
        diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        return albedo_add, diffuse_tint

    @property
    def state_size(self):
        return 2

    # For this state, we simply store its value and the robot_can_toggle steps.
    def _dump_state(self):
        return dict(value=self.value, hand_in_marker_steps=self.robot_can_toggle_steps)

    def _load_state(self, state):
        # Nothing special to do here when initialized vs. uninitialized
        self._set_value(state["value"])
        self.robot_can_toggle_steps = state["hand_in_marker_steps"]

    def serialize(self, state):
        return th.tensor([state["value"], state["hand_in_marker_steps"]], dtype=th.float32)

    def deserialize(self, state):
        return dict(value=bool(state[0]), hand_in_marker_steps=int(state[1])), 2
