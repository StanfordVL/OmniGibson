from collections import Counter, defaultdict
import json
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

import pymxs
rt = pymxs.runtime

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$")
SIDES =  {
    # Facing the object
    # "front": R.from_euler("xy", [-np.pi / 2, np.pi / 2]),

    # Top down
    # "top": R.from_euler("Z", -np.pi / 2),

    # Diagonal
    "diagonal": R.from_euler("xy", [-np.pi / 2, np.pi / 2]) * R.from_euler("XYZ", [0, -np.pi / 6, np.pi / 6]),

    # "top", "left", "right", "front", "back"
}
SUCCESS_FILENAME = "generate_object_images.success"

# The camera somehow has different axes.
def main():
    output_dir = os.path.join(rt.maxFilePath, "artifacts")
    images_dir = os.path.join(output_dir, "object_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Set the render preset
    preset_path = r"C:\Users\Cem\Documents\3ds Max 2022\renderpresets\imgrender.rps"
    preset_categories = rt.renderpresets.LoadCategories(preset_path)
    assert rt.renderpresets.Load(0, preset_path, preset_categories)

    # Set the exposure control
    ec = rt.VRay_Exposure_Control()
    ec.mode = 106
    ec.ev = -2
    rt.SceneExposureControl.exposurecontrol = ec

    # Group the objects
    failed = []
    groups = defaultdict(list)

    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        match = PATTERN.match(obj.name)
        if match is None:
            continue

        if int(match.group("instance_id")) != 0:
            continue

        groups[(match.group("category"), match.group("model_id"), match.group("instance_id"))].append(obj)

    for group_key, group in groups.items():
        try:
            # Find the base link obj
            matches = [PATTERN.match(obj.name) for obj in group]
            bases = [obj for obj, match in zip(group, matches) if match is not None and (not match.group("link_name") or match.group("link_name") == "base_link")]
            base, = bases

            # Hide everything
            for x in rt.objects:
                x.isHidden = True

            # Set the lights to be fixed for now
            for light in rt.lights:
                light.isHidden = True
            #     light.normalizeColor = 1
            #     light.multiplier = 15000

            # Unhide the objects in the group
            for x in group:
                x.isHidden = False

            # Get the object rotation
            obj_quat = base.rotation
            obj_rot = R.from_quat([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w])

            # Render
            for side_name, side_rot in SIDES.items():
                # Switch the side
                assert rt.viewport.setType(rt.Name("view_top")) 

                # Rotate the camera to match the object.
                camera_rot = side_rot * obj_rot.inv()
                rt.viewport.rotate(rt.Quat(*[float(x) for x in camera_rot.as_quat()]))

                # Set the zoom extent
                rt.select([])
                rt.select(group)
                assert pymxs.runtime.execute('max tool zoomextents')
                rt.forceCompleteRedraw()
                rt.windows.processPostedMessages()

                # Render the image
                obj_name = "-".join(group_key)
                output_filename = f"{obj_name}--{side_name}.png"
                image_path = os.path.abspath(os.path.join(images_dir, output_filename))
                if os.path.exists(image_path):
                    os.remove(image_path)
                print("Saving to", image_path)
                assert pymxs.runtime.render(outputFile=image_path)
        except:
            failed.append(group_key)

    success = True # (len(failed) == 0)
    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()