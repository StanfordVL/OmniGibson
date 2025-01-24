import omnigibson as og
cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
    },
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb"],
            "default_arm_pose": "diagonal30",
            "default_reset_mode": "tuck",
        },
    ],
    "task": {
        "type": "BehaviorTask",
        "activity_name": "laying_wood_floors",
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": True,
    },
}
env = og.Environment(configs=cfg)