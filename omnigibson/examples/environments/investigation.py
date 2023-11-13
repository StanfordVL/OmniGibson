#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pdb
import numpy as np
import omnigibson as og
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


# In[2]:


cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    }

from omnigibson.maps.segmentation_map import SegmentationMap
from omnigibson.utils.asset_utils import get_og_scene_path, get_available_og_scenes
from omni.isaac.core.utils.viewports import set_camera_view

# Get segmentation map
scene_path = get_og_scene_path("Rs_int")
seg_map = SegmentationMap(scene_path)

print(seg_map.room_sem_name_to_sem_id)
room_instance = "living_room_0"
center = None
if room_instance not in seg_map.room_ins_name_to_ins_id:
    print("room_instance [{}] does not exist.".format(room_instance))
    print(seg_map.room_ins_id_to_ins_name.items())
else:
    ins_id = seg_map.room_ins_name_to_ins_id[room_instance]
    room_pixel_coords = np.nonzero(seg_map.room_ins_map == ins_id)

    center = np.mean(room_pixel_coords, 1)
    print("CENTER:", center)

    print("ROOM PIXELS")
    print(room_pixel_coords)
    for x in room_pixel_coords:
        print(x.shape)

print("SEG_MAP")
print(seg_map.room_ins_map)
print(seg_map.room_ins_map.shape)



#Simulator must be stopped before loading scene!
og.sim.stop()

# Load the environment
env = og.Environment(configs=cfg)

# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()


# In[7]:
from omnigibson.utils.asset_utils import get_all_object_category_models, get_all_object_categories, decrypted

scene_objects = sorted([obj.name for obj in og.sim.scene.objects])
all_object_categories = get_all_object_categories()

import random

system_prompt = """

Imagine you're a sort of interior designer, but rather than physical spaces, you're designing a realistic and detailed indoor environment using a list of functions. You will receive an observation at the beginning and after each function call. Each function will allow you to place and arrange objects within the scene. Make sure the end result reflects both the complexity and functionality of a space that humans live and work in, with an appropriate mix of disorder and order. For instance, consider a function 'onTop', which allows you to place one object on top of another. Its parameters 'target_object1' and 'target_object2' represent the object being placed and the object it's being placed upon, respectively. Use this function and others like it to create a natural indoor scene. if a function call is successful in simulation, you will see a summary in the next human prompt at the end of the list of function calls you have already made. When you think the scene is sufficiently populated, stop responding with a function call.

""".strip()

#random.choices(all_object_categories, k=300)
human_prompt = f"""
The following is the list of all available objects that can be chosen as target_object1 to functions and after they are chosen, they will be imported into the scene in a way that creates the desired effect by the function called:
{random.choices(all_object_categories, k=300)}
The scene contains the below objects, which you can use as potential target_object2 to functions:
{scene_objects}

""".strip()

# In[18]:

functions = [
        
    {
        "name": "onTop",
        "description": "pick an object to be placed on top of another object in the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_object1": {
                    "type": "string",
                    "description": "The name of the object to place the on top of another object which will be defined as target_object2. You can pick this from the list of all available objects. target_object1 must be only an object category without model ID(6 random characters) or instance number.",
                },
                "target_object2": {
                    "type": "string",
                    "description": "The name of the object to place the object defined as the property \"target_object1\" on top of. Before calling this function, you need to have picked target_object1 first. You can pick \"target_object2\" from the list of all objects that the scene currently contains. These objects come in the form of categoryname_6randomcharacters_instancenumber",
                },
            },
            "required": ["target_object1", "target_object2"],
        },
    },
    {
        "name": "inside",
        "description": "pick an object to place inside another object that's in the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_object1": {
                    "type": "string",
                    "description": "The name of the object to place inside another object you will pick next. You can pick this from the list of all available objects. target_object1 must be only an object category without model ID(6 random characters) or instance number.",
                },
                "target_object2": {
                    "type": "string",
                    "description": "The name of the object to place the object denoted as target_object1 inside.  You can pick \"target_object2\" from the list of all objects that the scene currently contains.",
                },
            },
            "required": ["target_object1", "target_object2"],
        },
    },

]

# In[ ]:
import json
import textwrap
from pxr import Usd

def wrap_text(t):
    paragraphs = t.split("\n")
    return "\n".join(textwrap.fill(x, 80) for x in paragraphs)

def print_message(message):
    print(f"{message['role']}:")
    if message["content"]:
        print(textwrap.indent(wrap_text(message["content"]), "    "))
    elif message["function_call"]:
        fn_name = message["function_call"]["name"]
        fn_args = ", ".join(f"{k}={v}" for k, v in json.loads(message["function_call"]["arguments"]).items())
        print(textwrap.indent(f"Call function {fn_name}({fn_args})", "    "))
    print()

import openai
from omnigibson.utils.object_state_utils import  sample_kinematics
from omnigibson.objects.dataset_object import DatasetObject

def get_bbox(category, model):
    usd_path = DatasetObject.get_usd_path(category=category, model=model)
    usd_path = usd_path.replace(".usd", ".encrypted.usd")
    with decrypted(usd_path) as fpath:
        stage = Usd.Stage.Open(fpath)
        prim = stage.GetDefaultPrim()
        bbox = prim.GetAttribute("ig:nativeBB").Get()
    return bbox

openai.api_key = "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"
   
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": human_prompt},
]


for message in messages:
    print_message(message)

function_calls = []
# "gpt-3.5-turbo"
# "gpt-4-32k"

for _ in range(4):
    print("Stopping...")
    print(seg_map.room_sem_name_to_sem_id)
    room_instance = "living_room_0"
    if room_instance not in seg_map.room_ins_name_to_ins_id:
        print("room_instance [{}] does not exist.".format(room_instance))
        print(seg_map.room_ins_id_to_ins_name.items())
    else:
        ins_id = seg_map.room_ins_name_to_ins_id[room_instance]
        room_pixel_coords = np.nonzero(seg_map.room_ins_map == ins_id)
        print("ROOM PIXELS")
        print(room_pixel_coords)
        for x in room_pixel_coords:
            print(x.shape)
        center = np.mean(room_pixel_coords, 1)
        print("CENTER:", center)

    print("SEG_MAP")
    print(seg_map.room_ins_map)
    print(seg_map.room_ins_map.shape)
    eps = 0.0000001
    scene_path = get_og_scene_path("Rs_int")

    # add z -dimension
    camera_loc = np.concatenate((center, [10]))
    camera_target = np.concatenate((center + eps, [0]))

    set_camera_view(camera_loc, camera_target, "/World/viewer_camera")


    try:
        while True:
            og.sim.render()

    except KeyboardInterrupt:
        print("continuing...")

    print("Generating...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    print_message(response_message)

    if "function_call" not in response_message:
        break


    function_name = response_message["function_call"]["name"].upper()
    function_args = json.loads(response_message["function_call"]["arguments"])
    target_object1_category = function_args["target_object1"]
    if not get_all_object_category_models(target_object1_category):
        continue

    target_object1_model = random.choice(get_all_object_category_models(target_object1_category))
    current_object1_instances = [int(obj.name.strip("_")[-1]) for obj in og.sim.scene.objects if  f"{target_object1_category}_{target_object1_model}" in obj.name]
    target_object1_instanceID = max(current_object1_instances) + 1 if current_object1_instances else 0
    target_object1_name = f"{target_object1_category}_{target_object1_model}_{target_object1_instanceID}"
    target_object1 = DatasetObject(name = target_object1_name, category = target_object1_category, model = target_object1_model)

    bounding_box = get_bbox(target_object1_category, target_object1_model)
    if not np.all(np.array(bounding_box) < 0.5):
        print(bounding_box)
        continue
    og.sim.import_object(target_object1)
    target_object2 = [obj for obj in og.sim.scene.objects if obj.name == function_args["target_object2"]][0]
    for _ in range(20):
        env.step([])

    predicate = "onTop" if function_name.lower() == "ontop" else "inside"
    try:
        function_response = str(sample_kinematics(
            predicate,
            target_object1,
            target_object2
        ))
        
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        print_message(messages[-1])
        function_calls.append(f"""{target_object1_category} {predicate} {function_args["target_object2"]}""")
    except:
        og.sim.remove_object(target_object1)
        
    scene_objects = sorted([obj.name for obj in og.sim.scene.objects])

    human_prompt = f"""
    The following is the list of all available objects that can be chosen as target_object1 to functions and after they are chosen, they will be imported into the scene in a way that creates the desired effect by the function called:
    {random.choices(all_object_categories, k=300)}
    The scene contains the below objects, which you can use as potential target_object2 to functions:
    {scene_objects}
    Here is the list of function calls you have already made:
    {function_calls}

    """.strip()

    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": human_prompt},
    ]
# In[]

for _ in range(20):
    env.step([])

print(function_calls)

# In[]
while True:
    og.sim.render()
# %%
