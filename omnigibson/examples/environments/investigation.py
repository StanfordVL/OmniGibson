#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import omnigibson as og
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


# In[2]:


import os, yaml
#from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

# Load the pre-selected configuration and set the online_sampling flag
# config_filename = os.path.join(og.example_config_path, "basic_cfg.yaml")
# cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    }


#Simulator must be stopped before loading scene!
og.sim.stop()
# Load the environment
env = og.Environment(configs=cfg)
# env.robots[0]._links['base_link'].mass = 10000

#controller = StarterSemanticActionPrimitives(env.task, env.scene, env.robots[0], teleport=True)

# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()


# In[3]:


# task = env.task
# task_initial_conditions = env.task.activity_natural_language_initial_conditions
# task_goal_conditions = env.task.activity_natural_language_goal_conditions


# task_name = task.activity_name.replace("_", " ").capitalize()
# assert all(x.isalpha() or x.isspace() for x in task_name)

# def stringify_conds(conds):
#     string_conds = []
#     for c in conds:
#         for k, v in task.object_scope.items():
#             synset = k.split(".")[0]
#             num = k.split(".")[-1].split("_")[-1]
#             repl_key = synset+num
#             if not v.name:
#                 pdb.set_trace()
#             c = c.replace("roomroom", "room").replace(repl_key, v.name)
            
#         string_conds.append(f"- {c}")
        
#     return "\n".join(string_conds)

# initial_conditions = stringify_conds(task_initial_conditions)
# goal_conditions = stringify_conds(task_goal_conditions)


# In[6]:


rooms = "\n".join(sorted({
    f"- {rm}"
    for obj in env.scene.objects
    for rm in (obj.in_rooms if hasattr(obj, "in_rooms") and obj.in_rooms else [])
    if rm}))


# In[7]:

#TODO: The baseRobot part doesn't do any harm, but I think it is redundant; TODO: check
from omnigibson.robots import BaseRobot
# obj_room_pairs = {
#     (obj, (", ".join(obj.in_rooms) if hasattr(obj, "in_rooms") and obj.in_rooms else None))
#     for obj in env.scene.objects
#     if not isinstance(obj, BaseRobot)
# }
# objects = "\n".join(sorted(
#     f"- {obj.name} (in rooms {rm})" if rm else f"- {obj.name}"
#     for obj, rm in obj_room_pairs
# ))
objects = sorted([obj.name for obj in og.sim.scene.objects])

# In[8]:


# Use the scene graph API to get the current state of the scene
# def get_observation(env):
#     return "\n".join(f"- {state}({obj1.name}, {obj2.name})" for obj1, obj2, state in env.get_scene_graph().edges)
# initial_observation = get_observation(env)


# In[9]:


system_prompt = """

Imagine you're a sort of interior designer, but rather than physical spaces, you're designing a realistic and detailed indoor environment using a list of functions. You will receive an observation at the beginning and after each function call. Each function will allow you to place and arrange objects within the scene. Make sure the end result reflects both the complexity and functionality of a space that humans live and work in, with an appropriate mix of disorder and order. For instance, consider a function 'onTop', which allows you to place one object on top of another. Its parameters 'target_object1' and 'target_object2' represent the object being placed and the object it's being placed upon, respectively. Use this function and others like it to create a natural indoor scene.

""".strip()

#TODO: find a way to import objects into the scene; we want target_object1 to be from imported objects
#TODO: reflect this in the function definitions
# # In[10]:

# human_prompt initially contained the following three: 
# Your task: {task_name}
# Initial condition: all of the below are true at the beginning of the task.
# {initial_conditions}

# Goal condition: all of the below need to be true for the task to be complete:
# {goal_conditions}

# Observation: all of the below predicates are currently true:
# {initial_observation}
# The scene contains the below rooms:
# {rooms}

human_prompt = f"""
The scene contains the below objects, which you can use as arguments to functions:
{objects}
""".strip()


# In[18]:


# from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet

functions = [
        
    {
        "name": "onTop",
        "description": "pick an object to be placed on top of another object in the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_object1": {
                    "type": "string",
                    "description": "The name of the object to place the on top of another object which will be defined as target_object2.",
                },
                "target_object2": {
                    "type": "string",
                    "description": "The name of the object to place the object defined as the property \"target_object1\" on top of. Before calling this function, you need to have picked target_object1 first.",
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
                    "description": "The name of the object to place inside another object you will pick next.",
                },
                "target_object2": {
                    "type": "string",
                    "description": "The name of the object to place the object denoted as target_object1 inside.",
                },
            },
            "required": ["target_object1", "target_object2"],
        },
    },

]

import json

# def run_primitive(primitive_name, object_name):
#     try:
#         obj = env.scene.object_registry("name", object_name)
#         # assert primitive_name in StarterSemanticActionPrimitive, f"No such function {primitive_name} available."
#         prim_fn = controller.controller_functions[StarterSemanticActionPrimitiveSet[primitive_name]]
#         for action in prim_fn(obj):
#             env.step(action)

#         # Success scenario here
#         return json.dumps({
#             "result": f"Successfully called {primitive_name}({object_name})",
#             "new_observation": get_observation(env)
#         })
#     except Exception as e:
#         return json.dumps({"result": str(e)})


# In[ ]:
import textwrap
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

openai.api_key = "ADD_KEY"
   
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": human_prompt},
]

for message in messages:
    print_message(message)


for _ in range(30):
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
    function_response = sample_kinematics(
        function_name,
        function_args["target_object"][0],
        function_args["target_object"][1]
    )
    
    messages.append(response_message)
    messages.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        }
    )  # extend conversation with function response
    print_message(messages[-1])




# %%
