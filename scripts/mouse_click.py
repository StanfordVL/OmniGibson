import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.grasping_planning_utils import get_orientation_facing_vector_with_random_yaw


def add_click_callback(callback, window_name="Viewport"):
    def _recursive_find_sceneview(frame):
        if "SceneView" in str(type(frame)):
            return frame

        for child in lazy.omni.ui.Inspector.get_children(frame):
            sceneview = _recursive_find_sceneview(child)
            if sceneview:
                return sceneview

        return None

    class ClickGesture(lazy.omni.ui.scene.ClickGesture):
        def __init__(self, viewport_api, callback, mouse_button: int = 0):
            super().__init__(mouse_button=mouse_button)
            self.__viewport_api = viewport_api
            self._callback = callback

        def on_ended(self, *args):
            self.__ndc_mouse = self.sender.gesture_payload.mouse
            mouse, viewport_api = self.__viewport_api.map_ndc_to_texture_pixel(self.__ndc_mouse)
            if mouse and viewport_api:
                viewport_api.request_query(mouse, self._callback)

    class ViewportClickManipulator(lazy.omni.ui.scene.Manipulator):
        def __init__(self, viewport_api, callback, mouse_button: int = 0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__gesture = ClickGesture(viewport_api, callback, mouse_button)
            self.__transform = None

        def on_build(self):
            # Need to hold a reference to this or the sc.Screen would be destroyed when out of scope
            self.__transform = lazy.omni.ui.scene.Transform()
            with self.__transform:
                self.__screen = lazy.omni.ui.scene.Screen(gesture=self.__gesture)

        def destroy(self):
            if self.__transform:
                self.__transform.clear()
                self.__transform = None
            self.__screen = None

    viewport_api, viewport_window = lazy.omni.kit.viewport.utility.get_active_viewport_and_window(
        window_name=window_name
    )
    scene_view = _recursive_find_sceneview(viewport_window.frame)
    with scene_view.scene:
        ViewportClickManipulator(viewport_api, callback)


def main():
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["walls", "floors"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["scan", "rgb", "depth"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "controller_config": {
                    "base": {
                        "name": "DifferentialDriveController",
                    },
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "command_input_limits": "default",
                        "command_output_limits": [[-0.2, -0.2, -0.2, -0.5, -0.5, -0.5], [0.2, 0.2, 0.2, 0.5, 0.5, 0.5]],
                        "mode": "pose_absolute_ori",
                        "kp": 300.0,
                    },
                    "gripper_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                    },
                    "camera": {"name": "JointController", "use_delta_commands": False},
                },
            }
        ],
    }
    env = og.Environment(cfg)

    def callback(prim_path, pos_3d, pos_2d):
        direction = np.array(pos_3d) - og.sim.viewer_camera.get_position()
        hit = og.sim.psqi.raycast_closest(
            origin=og.sim.viewer_camera.get_position(),
            dir=direction / np.linalg.norm(direction),
            distance=np.linalg.norm(direction) * 1.5,
            bothSides=True,
        )
        if not hit["hit"]:
            return
        position = np.array(hit["position"])
        normal = np.array(hit["normal"])
        prim_path = hit["rigidBody"]
        distance = hit["distance"]
        grasp_quat = get_orientation_facing_vector_with_random_yaw(normal)

    add_click_callback(callback)

    while True:
        og.sim.render()


if __name__ == "__main__":
    main()
