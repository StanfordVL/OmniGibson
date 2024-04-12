import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy


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
        }
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
        print(hit)

    add_click_callback(callback)

    while True:
        og.sim.render()


if __name__ == "__main__":
    main()
