import logging
from sys import platform

import numpy as np
import pybullet as p
from PIL import Image
import cv2

from scipy.spatial.transform import Rotation as R

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_ig_scenes
from igibson.utils.derivative_dataset.generators import uniform_generator, gaussian_target_generator
from igibson.utils.derivative_dataset.pertubers import texture_randomization, object_randomization

RENDER_WIDTH = 1080
RENDER_HEIGHT = 720

PERTURBERS = [
    lambda env: None,
    texture_randomization,
    object_randomization
    # igibson.utils.derivative_dataset.perturbers.joint_randomization,
    # igibson.utils.derivative_dataset.perturbers.light_randomization,
    # igibson.utils.derivative_dataset.perturbers.clutter_sampling,
    ]



GENERATORS = [
    uniform_generator,
    # gaussian_target_generator
    # igibson.utils.derivative_dataset.generators.robot_pov_generator,
    # igibson.utils.derivative_dataset.generators.object_focused_generator,
    # igibson.utils.derivative_dataset.generators.,
]

FILTERS = [
    # igibson.utils.derivative_dataset.filters.too_close_to_object
    # igibson.utils.derivative_dataset.filters.too_high
]




def main( headless=True, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    available_ig_scenes = get_available_ig_scenes()
    scene_id = available_ig_scenes[12]
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=RENDER_WIDTH, image_height=RENDER_HEIGHT, vertical_fov=60,
        rendering_settings=settings
    )
    perturber = PERTURBERS[2]   # TODO: Pick randomly
    scene = InteractiveIndoorScene(
        scene_id,
        build_graph=True,
        texture_randomization = True,
        object_randomization=True
    )
    s.import_scene(scene)

    #scene = simulator.scene

    for _ in range(100):
        s.step()


    for generator in GENERATORS:
        for i in range(10):
            perturber(scene)
            camera_pos, camera_target, camera_up = generator(scene)

            if any(filter(s, camera_pos, camera_target, camera_up) for filter in FILTERS):
                continue

            renderer : MeshRenderer = s.renderer
            s.renderer.set_camera(camera_pos, camera_target, camera_up)
            rgb, segmask = s.renderer.render(('rgb', 'seg'))

            rgb_img = Image.fromarray(np.uint8(rgb[:, :, :3] * 255))
            rgb_img.save(f'/scr/monaavr/test_img/rgbtest{i}.png')
            segmask = Image.fromarray(np.uint8(segmask[..., 0] * 255))
            segmask.save(f'/scr/monaavr/test_img/segmasktest{i}.png')

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()