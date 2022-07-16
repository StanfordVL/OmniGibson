import matplotlib.pyplot as plt

from igibson import app, ig_dataset_path, Simulator
from igibson.utils.usd_utils import CollisionAPI
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.robots import Tiago
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import igibson.utils.transform_utils as T
import json
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim, set_prim_property, get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
import omni
import carb
from igibson.sensors.vision_sensor import VisionSensor
from pxr import Gf
import os
from PIL import Image
from pathlib import Path
import datetime


##### SET THIS ######
SCENE_ID = "gates_apartment"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/{SCENE_ID}/urdf/{SCENE_ID}_best_template.usd"
IMG_HEIGHT = 720
IMG_WIDTH = 1280
#(array([2.37998462, 0.13371089, 0.01527086]), array([4.72678607e-08, 5.57332172e-08, 7.60978907e-02, 9.97100353e-01]))

ROBOT_POS = np.array([2.0489, -0.18226, 0.01527086])
ROBOT_QUAT = T.euler2quat(np.array([0,0,4.084]))

# ROBOT_POS = np.array([0.5318, 1.3472, 0.01527086])
# ROBOT_QUAT = T.euler2quat(np.array([0,0,2.7])) #np.array([4.72678607e-08, 5.57332172e-08, 7.60978907e-02, 9.97100353e-01])

PHOTO_SAVE_DIRECTORY = f"/scr/robertom/og_photos/{SCENE_ID}"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

HFOV = 58 * np.pi / 180

sim = Simulator()

# Load scene
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
sim.import_scene(scene=scene)

cam = VisionSensor(
    prim_path="/World/viewer_camera",
    name="camera",
    modalities=["rgb", "depth"],
    image_height=IMG_HEIGHT,
    image_width=IMG_WIDTH,
    # viewport_name="Viewport",
)
cam.set_attribute("clippingRange", Gf.Vec2f(0.001, 10000000.0))
cam.set_attribute("focalLength", 17.0)
cam.visible = False
sim.step()
cam.visible = True
cam.initialize()
sim.step()
cam.image_width = IMG_WIDTH
cam.image_height = IMG_HEIGHT
sim.step()
sim.step()

sim.step()
sim.stop()

def set_lights(intensity):
    world = get_prim_at_path("/World")
    for prim in world.GetChildren():
        for prim_child in prim.GetChildren():
            for prim_child_child in prim_child.GetChildren():
                if "Light" in prim_child_child.GetPrimTypeInfo().GetTypeName():
                    # print("Modifying light!")
                    prim_child_child.GetAttribute("intensity").Set(intensity)

    for i in range(20):
        sim.step()

def set_pt():
    app.config["renderer"] = "PathTracing"
    app._set_render_settings()
    for i in range(30):
        sim.step()

def set_rt():
    app.config["renderer"] = "RayTracingLighting"
    app.config["anti_aliasing"] = 1
    app._set_render_settings()
    for i in range(10):
        sim.step()


set_rt()

default_light = 8e4
set_lights(default_light)

def take_photo(use_rt=None, name=f"{SCENE_ID}", rootdir=PHOTO_SAVE_DIRECTORY):
    # os.makedirs(os.path.dirname(rootdir), exist_ok=True)
    if use_rt is not None:
        if use_rt:
            set_rt()
        else:
            set_pt()
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    #img = cam.get_obs()["rgb"][:, :, :3]
    
    # Robot observations
    obs = robot.get_obs()
    img = obs["robot:eyes_Camera_sensor_depth"]
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    print(img.shape)
    img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
    print(img.shape)
    print(img)
    print(repr(np.array(obs['robot:base_front_laser_link_Lidar_sensor_scan']).flatten()))
    print(repr(np.array(obs['robot:base_rear_laser_link_Lidar_sensor_scan']).flatten()))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Image.fromarray(img).save(f"{rootdir}/{name}_{app.config['renderer']}_{timestamp}_depth.png")
    img = obs["robot:eyes_Camera_sensor_rgb"][:, :, :3]
    Image.fromarray(img).save(f"{rootdir}/{name}_{app.config['renderer']}_{timestamp}_rgb.png")
    
    if use_rt is not None:
        set_rt()


class CameraMover:
    def __init__(self, cam, delta=0.25):
        self.cam = cam
        self.delta = delta
        self.light_val = default_light

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def change_light(self, delta):
        self.light_val += delta
        set_lights(self.light_val)

    def print_info(self):
        print("*" * 40)
        print("CameraMover! Commands:")
        print()
        print(f"\t Right Click + Drag: Rotate camera")
        print(f"\t W / S : Move camera forward / backward")
        print(f"\t A / D : Move camera left / right")
        print(f"\t T / G : Move camera up / down")
        print(f"\t U / J : Increase / decrease the lights")
        print(f"\t 1 : Take Rayracing photo")
        print(f"\t 2 : Take Pathtracing photo")



    @property
    def input_to_function(self):
        return {
            carb.input.KeyboardInput.SPACE: lambda: take_photo(use_rt=True),
            carb.input.KeyboardInput.KEY_1: lambda: take_photo(use_rt=True),
            carb.input.KeyboardInput.KEY_2: lambda: take_photo(use_rt=False),
            carb.input.KeyboardInput.U: lambda: self.change_light(delta=2e4),
            carb.input.KeyboardInput.J: lambda: self.change_light(delta=-2e4),
        }

    @property
    def input_to_command(self):
        return {
            carb.input.KeyboardInput.D: np.array([self.delta, 0, 0]),
            carb.input.KeyboardInput.A: np.array([-self.delta, 0, 0]),
            carb.input.KeyboardInput.W: np.array([0, 0, -self.delta]),
            carb.input.KeyboardInput.S: np.array([0, 0, self.delta]),
            carb.input.KeyboardInput.T: np.array([0, self.delta, 0]),
            carb.input.KeyboardInput.G: np.array([0, -self.delta, 0]),
            carb.input.KeyboardInput.Z: np.array([np.pi/8, 0]),
            carb.input.KeyboardInput.X: np.array([-np.pi/8, 0]),
            carb.input.KeyboardInput.C: np.array([0, np.pi/8]),
            carb.input.KeyboardInput.V: np.array([ 0, -np.pi/8]),
            carb.input.KeyboardInput.B: np.array([np.pi / 8]),
            carb.input.KeyboardInput.N: np.array([-np.pi / 8]),
        }

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events
        Args:
            event (int): keyboard event type
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                or event.type == carb.input.KeyboardEventType.KEY_REPEAT:

            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input in self.input_to_function:
                self.input_to_function[event.input]()

            else:
                command = self.input_to_command.get(event.input, None)

                if command is not None:
                    if len(command) == 3:
                        # Convert to world frame to move the camera
                        transform = T.quat2mat(self.cam.get_orientation())
                        delta_pos_global = transform @ command
                        self.cam.set_position(self.cam.get_position() + delta_pos_global)
                        print(self.cam.get_position())
                    elif len(command) == 2:
                        euler = T.quat2euler(self.cam.get_orientation())
                        euler[1] += command[0]
                        euler[2] += command[1]
                        quat = T.euler2quat(euler)
                        self.cam.set_orientation(quat)
                        print(self.cam.get_orientation())
                    else:
                        euler = T.quat2euler(self.cam.get_orientation())
                        euler[0] += command[0]
                        quat = T.euler2quat(euler)
                        self.cam.set_orientation(quat)
                        print(self.cam.get_orientation())

        return True

cam_mover = CameraMover(cam=cam)

# Move lamp
lamp = sim.scene.object_registry("name", "table_lamp_bbentu_0").set_position([2.8367, 1.53104, 1.0504])

# Import robot
robot = Tiago(prim_path=f"/World/robot", name="robot", obs_modalities=["rgb", "depth", "scan", "scan_rear"])
sim.import_object(obj=robot)

robot.set_position_orientation(ROBOT_POS, ROBOT_QUAT)
cam_init_pos = np.array([0.21338835, 3.41132297, 1.66603275])
cam_init_ori = np.array([-0.10838638,  0.54489511 , 0.81549316, -0.16221167])
cam.set_position_orientation(cam_init_pos, cam_init_ori)

robot.sensors["robot:eyes_Camera_sensor"].image_width = 640
robot.sensors["robot:eyes_Camera_sensor"].image_height = 480
# robot.sensors['robot:base_rear_laser_link_Lidar_sensor_scan']

cam = get_prim_at_path("/World/robot/eyes/Camera")
h_aperture = cam.GetAttribute("horizontalAperture").Get()
cam.GetAttribute("focalLength").Set((h_aperture / 2.0) / np.tan(HFOV / 2.0))

for lidar_name in ["front", "rear"]:
    lidar = get_prim_at_path("/World/robot/base_{}_laser_link/Lidar".format(lidar_name))
    lidar.GetAttribute("drawLines").Set(True)
    # lidar.GetAttribute("drawPoints").Set(True)

    lidar.GetAttribute("minRange").Set(0.13)
    lidar.GetAttribute("horizontalFov").Set(190.3144102778604)
    lidar.GetAttribute("horizontalResolution").Set(0.3332999980051271)
    # lidar.GetAttribute("yawOffset").Set(-45)
    lidar.GetRelationship("physics:filteredPairs").AddTarget("/World/robot/base_link")

sim.play()
robot.reset()
robot.keep_still()
sim.step()
sim.pause()

cam_mover.print_info()


from IPython import embed
# embed()
for i in range(100000):
    sim.step()

app.close()
