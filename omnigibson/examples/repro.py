import pathlib
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading

asset_path = str(pathlib.Path(__file__).parent / "bed.usda")
simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/World")
create_prim("/DistantLight", "DistantLight")
# wait for things to load
simulation_app.update()
while is_stage_loading():
    simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
simulation_context.play()

try:
    step = 0
    while True:
        step += 1
        if step % 100 == 0:
            print("step", step)  
        simulation_context.step(render=True)
except KeyboardInterrupt:
    pass

simulation_context.stop()
simulation_app.close()