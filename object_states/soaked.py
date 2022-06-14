from igibson import Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omni.isaac.core.utils.prims import get_prim_at_path
import igibson
from igibson.objects.dataset_object import DatasetObject
import numpy as np
import os

# Create environment
sim = Simulator()
	
usd_path = os.path.join(igibson.ig_dataset_path, "scenes/Rs_int/urdf/Rs_int_best_template.usd")
scene = InteractiveTraversableScene(
    scene_model="Rs_int",
    usd_path=usd_path,
    load_room_types=["kitchen"]
)
sim.import_scene(scene)
ceiling = sim.scene.object_registry("name", "ceilings")
ceiling.visible = False

camera = get_prim_at_path("/World/viewer_camera")
camera.GetAttribute("xformOp:translate").Set((0.17, 1.34, 2.49))
camera.GetAttribute("xformOp:rotateXYZ").Set((51, 0, 34))

paper_towel = os.path.join(igibson.ig_dataset_path, "objects/paper_towel/33_0/usd/33_0.usd")
paper_towel_obj = DatasetObject(
    prim_path="/World/paper_towel",
    usd_path=paper_towel,
    name="paper_towel_000_1",
    abilities={"soakable": {}, "cleaningTool": {}},
)
sim.import_object(paper_towel_obj)

sim.play()
idx = 0
while True:
    sim.step()
    idx += 1
    # breakpoint()
    # for obj in sim.scene.object_registry("states", igibson.object_states.ToggledOn):
    #     print(obj.name)
    if idx > 500:
        print("Toggling on sink!")
        sink = sim.scene.object_registry("name", "sink_35")
        sink.states[igibson.object_states.ToggledOn].set_value(True)
        sink = sim.scene.object_registry("name", "sink_42")
        sink.states[igibson.object_states.ToggledOn].set_value(True)

# print('done')
