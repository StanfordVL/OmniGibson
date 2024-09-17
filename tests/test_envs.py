import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm


@pytest.mark.parametrize("pipeline_mode", ["cpu", "cuda"], indirect=True)
class TestTasks:
    def task_tester(self, task_type, pipeline_mode):
        cfg = {
            "env": {
                "device": pipeline_mode,
            },
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": "Rs_int",
                "load_object_categories": ["floors", "breakfast_table"],
            },
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": [],
                }
            ],
            # Task kwargs
            "task": {
                "type": task_type,
                # BehaviorTask-specific
                "activity_name": "laying_wood_floors",
                "online_object_sampling": True,
            },
        }

        if og.sim is None:
            gm.ENABLE_OBJECT_STATES = True
            gm.ENABLE_TRANSITION_RULES = False
        else:
            # Make sure sim is stopped
            og.sim.stop()

        # Create the environment
        env = og.Environment(configs=cfg)

        env.reset()
        for _ in range(5):
            env.step(th.tensor(env.robots[0].action_space.sample(), dtype=th.float32))

        # Clear the sim
        og.clear()

    def test_dummy_task(self, pipeline_mode):
        self.task_tester("DummyTask", pipeline_mode)

    def test_point_reaching_task(self, pipeline_mode):
        self.task_tester("PointReachingTask", pipeline_mode)

    def test_point_navigation_task(self, pipeline_mode):
        self.task_tester("PointNavigationTask", pipeline_mode)

    def test_behavior_task(self, pipeline_mode):
        self.task_tester("BehaviorTask", pipeline_mode)


@pytest.mark.parametrize("pipeline_mode", ["cpu", "cuda"], indirect=True)
def test_rs_int_full_load(pipeline_mode):
    cfg = {
        "env": {
            "device": pipeline_mode,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
            }
        ],
        # Task kwargs
        "task": {
            "type": "DummyTask",
        },
    }

    # Make sure sim is stopped
    if og.sim:
        og.sim.stop()

    # Create the environment
    env = og.Environment(configs=cfg)

    env.reset()
    for _ in range(5):
        env.step(th.tensor(env.robots[0].action_space.sample(), dtype=th.float32))

    # Clear the sim
    og.clear()
