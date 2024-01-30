import pathlib
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading

from omni.isaac.core.prims import RigidPrimView

from scipy.spatial.transform import Rotation as R
import numpy as np

asset_path = str(pathlib.Path(__file__).parent / "bed.usda")
simulation_context = SimulationContext()

create_prim("/DistantLight", "DistantLight")
# wait for things to load
simulation_app.update()
while is_stage_loading():
    simulation_app.update()

bed_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
simulation_context.play()

view = RigidPrimView("/World/bed/base_link")
view.initialize()

try:
    print("-------------------------------")
    print("If we rotate an object such that it will not be perfectly stable after rotation,")
    print("e.g. it will fall down even if slightly after rotation, then the object renders properly.")
    input("Hit enter to try for 3k steps, and manually turn on PVD's collision mesh visualization while this runs.")
    step = 0
    for _ in range(3000):
        step += 1
        if step % 100 == 0:
            print("step", step)
            poses = view.get_world_poses()
            current_orientation_quat = poses[1][0][[1, 2, 3, 0]]
            current_rot = R.from_quat(current_orientation_quat)
            new_rot = R.from_euler('xyz', [0.01, 0.01, np.pi / 4]) * current_rot
            new_orn_quat = new_rot.as_quat()
            view.set_world_poses(orientations=[new_orn_quat[[3, 0, 1, 2]]])
            
        simulation_context.step(render=True)
        
    print("-------------------------------")
    print("However, if we only rotate around the z-axis, the object doesn't fall after rotating.")
    print("In this case, the PhysX transform (as seen through PVD) still updates but Omni renders")
    print("the object as if it's still in the original orientation.")
    input("Hit enter to try for 1k steps.")
    while True:
        step += 1
        if step % 100 == 0:
            print("step", step)
            poses = view.get_world_poses()
            current_orientation_quat = poses[1][0][[1, 2, 3, 0]]
            current_rot = R.from_quat(current_orientation_quat)
            new_rot = R.from_euler('z', np.pi / 4) * current_rot
            new_orn_quat = new_rot.as_quat()
            view.set_world_poses(orientations=[new_orn_quat[[3, 0, 1, 2]]])
            
        simulation_context.step(render=True)
except KeyboardInterrupt:
    pass

simulation_context.stop()
simulation_app.close()