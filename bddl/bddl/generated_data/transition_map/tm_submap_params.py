TM_SUBMAPS_TO_PARAMS = {
    "electric_mixer": {
        "input_objects": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "output_objects": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"
        }
    },
    "heat_cook": {
        "input_objects": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "container": {
            "required": False,
            "type": "synset"
        },
        # "container_input_relation": {
        #     "required"
        # },
        "heat_source": {
            "required": False,
            "type": "synset"
        },
        "output_objects": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"
        },
        # "container_output_relation": {
        #     "required"
        # },
        "timesteps": {
            "required": False,
            "type": "integer"
        }
    },
    "mixing_stick": {
        "input_objects": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "output_objects": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"

        }
    },
    "single_toggleable_machine": {
        "input_objects": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "machine": {
            "required": True,
            "type": "synset"

        },
        "output_objects": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"
        }
    }
}

