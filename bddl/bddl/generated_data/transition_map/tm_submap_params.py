TM_SUBMAPS_TO_PARAMS = {
    "electric_mixer": {
        "input_objects": {
            "required"
        },
        "output_objects": {
            "required"
        }
    },
    "heat_cook": {
        "input_objects": {
            "required"
        },
        "container": {
            "optional"
        },
        "container_input_relation": {
            "required"
        },
        "heat_source": {
            "optional"
        },
        "output_objects": {
            "required"
        },
        "container_output_relation": {
            "required"
        },
        "timesteps": {
            "optional"
        }
    },
    "mixing_stick": {
        "input_objects": {
            "required"
        },
        "output_objects": {
            "required"
        }
    },
    "single_toggleable_machine": {
        "input_objects": {
            "required"
        },
        "machine": {
            "required"
        },
        "output_objects": {
            "required"
        }
    },
    # "water_cook": {
    #     "timesteps": {
    #         "optional"
    #     }
    # },
    # "water_cook_ricecooker": {
    #     "machine": {
    #         "optional"
    #     },
    #     "timesteps": {
    #         "optional"
    #     }
    # }
}

