import logging
import os
import yaml
import copy

import omnigibson as og
from IPython import embed

def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment from a pre-defined configuration file.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Load the pre-selected configuration
    config_filename = os.path.join(og.example_config_path, "fetch_behavior_dummy.yaml")

    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    scenes_to_activities = {
        "Rs_int": [
            "putting_away_Halloween_decorations",
            "cleaning_sneakers",
            "testing_filled",
        ]
    }

    for scene_model, activities in scenes_to_activities.items():
        cfg["scene"]["scene_model"] = scene_model
        env = og.Environment(configs=cfg)
        scene_initial_state = copy.deepcopy(env.scene._initial_state)
        og.sim.stop()

        env.task_config["type"] = "BehaviorTask"
        env.task_config["online_object_sampling"] = True
        env.task_config["load_clutter"] = False

        for activity_name in activities:
            assert og.sim.is_stopped()
            env.task_config["activity_name"] = activity_name
            env._load_task()
            assert og.sim.is_stopped()

            print("finish")
            embed()

            og.sim.play()
            # This will actually reset the objects to their sample poses
            og.sim.scene.reset()

            # usd_file = "{}_task_{}_{}_{}_fixed_furniture_template".format(
            #     env.scene_config["scene_model"],
            #     env.task_config["activity_name"],
            #     env.task_config["activity_definition_id"],
            #     env.task_config["activity_instance_id"]
            # )
            # usd_file_path = os.path.join(og.og_dataset_path, "scenes", og.sim.scene.scene_model, "usd", usd_file + ".usd")
            #
            # for obj_scope, obj in env.task.object_scope.items():
            #     obj._prim.CreateAttribute("ig:objectScope", VT.String)
            #     obj._prim.GetAttribute("ig:objectScope").Set(obj_scope)
            #
            usd_file_path = f"/cvgl2/u/chengshu/Downloads/{activity_name}.usd"
            og.sim.save(usd_file_path)

            assert og.sim.is_stopped()

            for obj in env.task.sampled_objects:
                # This will remove the gym shoes
                og.sim.remove_object(obj)

            og.sim.play()
            # This will clear out the previous attachment group in macro particle systems
            og.sim.scene.load_state(scene_initial_state)
            og.sim.stop()

        og.sim.clear()

if __name__ == "__main__":
    main()