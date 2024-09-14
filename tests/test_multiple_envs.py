import pytest
import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition


@pytest.mark.parametrize("pipeline_mode", ["cpu", "cuda"], indirect=True)
class TestMultiEnvironment:
    def setup_multi_environment(self, pipeline_mode, num_of_envs, additional_objects_cfg=[]):
        cfg = {
            "env": {
                "device": pipeline_mode,
            },
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": "Rs_int",
                "load_object_categories": ["floors", "walls"],
            },
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": [],
                }
            ],
        }

        cfg["objects"] = additional_objects_cfg

        if og.sim is None:
            gm.RENDER_VIEWER_CAMERA = False
            gm.ENABLE_OBJECT_STATES = True
            gm.ENABLE_TRANSITION_RULES = False
        else:
            # Make sure sim is stopped
            og.sim.stop()

        vec_env = og.VectorEnvironment(num_of_envs, cfg)
        return vec_env

    def test_multi_scene_dump_and_load(self, pipeline_mode):
        vec_env = self.setup_multi_environment(pipeline_mode, 3)
        robot_displacement = th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
        scene_three_robot = vec_env.envs[2].scene.robots[0]
        robot_new_pos = scene_three_robot.get_position() + robot_displacement
        scene_three_robot.set_position(robot_new_pos)
        scene_three_state = vec_env.envs[2].scene._dump_state()
        og.clear()

        vec_env = self.setup_multi_environment(pipeline_mode, 3)
        initial_robot_pos_scene_one = vec_env.envs[0].scene.robots[0].get_position()
        vec_env.envs[0].scene._load_state(scene_three_state)
        new_robot_pos_scene_one = vec_env.envs[0].scene.robots[0].get_position()
        assert th.allclose(new_robot_pos_scene_one - initial_robot_pos_scene_one, robot_displacement, atol=1e-3)

        og.clear()

    def test_multi_scene_displacement(self, pipeline_mode):
        vec_env = self.setup_multi_environment(pipeline_mode, 3)
        robot_0_pos = vec_env.envs[0].scene.robots[0].get_position()
        robot_1_pos = vec_env.envs[1].scene.robots[0].get_position()
        robot_2_pos = vec_env.envs[2].scene.robots[0].get_position()

        dist_0_1 = robot_1_pos - robot_0_pos
        dist_1_2 = robot_2_pos - robot_1_pos
        assert th.allclose(dist_0_1, dist_1_2, atol=1e-3)
        og.clear()

    def test_multi_scene_scene_prim(self, pipeline_mode):
        vec_env = self.setup_multi_environment(pipeline_mode, 1)
        original_robot_pos = vec_env.envs[0].scene.robots[0].get_position()
        scene_state = vec_env.envs[0].scene._dump_state()
        scene_prim_displacement = th.tensor([10.0, 0.0, 0.0], dtype=th.float32)
        original_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position()
        vec_env.envs[0].scene.set_position_orientation(position=original_scene_prim_pos + scene_prim_displacement)
        vec_env.envs[0].scene._load_state(scene_state)
        new_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position()
        new_robot_pos = vec_env.envs[0].scene.robots[0].get_position()
        assert th.allclose(new_scene_prim_pos - original_scene_prim_pos, scene_prim_displacement, atol=1e-3)
        assert th.allclose(new_robot_pos - original_robot_pos, scene_prim_displacement, atol=1e-3)

    def test_multi_scene_particle_source(self, pipeline_mode):
        sink_cfg = dict(
            type="DatasetObject",
            name="sink",
            category="sink",
            model="egwapq",
            bounding_box=[2.427, 0.625, 1.2],
            abilities={
                "toggleable": {},
                "particleSource": {
                    "conditions": {
                        "water": [
                            (ParticleModifyCondition.TOGGLEDON, True)
                        ],  # Must be toggled on for water source to be active
                    },
                    "initial_speed": 0.0,  # Water merely falls out of the spout
                },
                "particleSink": {
                    "conditions": {
                        "water": [],  # No conditions, always sinking nearby particles
                    },
                },
            },
            position=[0.0, -1.5, 0.42],
        )

        vec_env = self.setup_multi_environment(pipeline_mode, 3, additional_objects_cfg=[sink_cfg])

        for env in vec_env.envs:
            sink = env.scene.object_registry("name", "sink")
            assert sink.states[object_states.ToggledOn].set_value(True)

        for _ in range(50):
            og.sim.step()
