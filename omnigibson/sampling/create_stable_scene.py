import logging
import os
import yaml
import copy
import time
import argparse
import bddl
import pkgutil
import omnigibson as og
from omnigibson.macros import gm, macros
import json
import csv
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Contains
from omnigibson.tasks import BehaviorTask
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear, get_system, MicroPhysicalParticleSystem
from omnigibson.systems.system_base import clear_all_systems, PhysicalParticleSystem
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.python_utils import create_object_from_init_info
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from omnigibson.utils.constants import PrimType
from bddl.activity import Conditions, evaluate_state
from utils import *
import numpy as np
import random


# TODO:
# 1. Set boundingCube approximation earlier (maybe right after importing the scene objects). Otherwise after loading the robot, we will elapse one physics step
# 2. Enable transition rule and refresh all rules before online validation

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None,
                    help="Scene model to sample tasks in")

gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

# macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0

def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # Parse arguments based on whether values are specified in os.environ
    # Priority is:
    # 1. command-line args
    # 2. environment level variables
    if args.scene_model is None:
        # This MUST be specified
        assert os.environ.get("SAMPLING_SCENE_MODEL"), "scene model MUST be specified, either as a command-line arg or as an environment variable!"
        args.scene_model = os.environ["SAMPLING_SCENE_MODEL"]

    # If we want to create a stable scene config, do that now
    default_scene_fpath = f"{gm.DATASET_PATH}/scenes/{args.scene_model}/json/{args.scene_model}_stable.json"
    if not os.path.exists(default_scene_fpath):
        create_stable_scene_json(scene_model=args.scene_model)

if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
