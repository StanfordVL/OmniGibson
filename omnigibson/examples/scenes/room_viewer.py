import omnigibson as og
from omnigibson.macros import gm

import numpy as np
import matplotlib.pyplot as plt

# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main():
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["walls", "floors"]
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)
    camera = og.sim.viewer_camera

    # Compute the horizontal and vertical FOV slope of the camera
    inv_int = np.linalg.inv(camera.intrinsic_matrix)
    topleft = inv_int @ np.array([0, 0, 1])
    bottomright = inv_int @ np.array([camera.image_width, camera.image_height, 1])
    horizontal_cover_per_meter = (bottomright[0] - topleft[0])
    vertical_cover_per_meter = (bottomright[1] - topleft[1])
    cover_per_meter = np.array([horizontal_cover_per_meter, vertical_cover_per_meter])

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Get the segmap object and the actual instance segmap
    scene = env.scene
    segmap = scene.seg_map
    insseg = segmap.room_ins_map

    # Visualize the rooms one by one
    for room in segmap.room_ins_name_to_ins_id:
        room_id = segmap.room_ins_name_to_ins_id[room]

        # Get the pixels that map to that room
        if not np.any(insseg == room_id):
            continue
        room_pixels = np.array(np.nonzero(insseg == room_id)).T
        room_pixels_in_world = segmap.map_to_world(room_pixels)
        center = np.mean(room_pixels_in_world, axis=0)
        max_distances = np.max(np.abs(room_pixels_in_world - center[None, :]), axis=0) + 2  # small margin
        camera_height = np.max(max_distances / cover_per_meter)
        camera_pos = np.array([*center, camera_height])

        # Move the camera there and keep rendering
        camera.set_position_orientation(camera_pos, [0, 0, 0, 1])

        # Render from the camera now. This needs to be repeated a bit
        for _ in range(3):
            og.sim.render()
        rgb = camera.get_obs()["rgb"][:, :, :3]
        plt.title(room)
        plt.imshow(rgb)
        plt.show()

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
