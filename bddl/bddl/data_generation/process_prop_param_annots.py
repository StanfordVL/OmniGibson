import json
import os
import pandas as pd
import pathlib


SYNS_TO_PROPS = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_canonical.json"
PROP_PARAM_ANNOTS_DIR = pathlib.Path(__file__).parents[1] / "generated_data" / "prop_param_annots"
PARAMS_OUTFILE_FN = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_params_new.json"


def create_get_save_propagated_annots_params():
    with open(SYNS_TO_PROPS, "r") as f:
        syns_to_props = json.load(f)
    
    for prop_fn in os.listdir(PROP_PARAM_ANNOTS_DIR):
        prop = prop_fn.split(".")[0]

        if prop == "particleRemover":
            # TODO 
            pass
        else:        
            param_annots = pd.read_csv(os.path.join(PROP_PARAM_ANNOTS_DIR, prop_fn)).to_dict(orient="records")
            for param_record in param_annots: 
                for param_name, param_value in param_record.items():
                    if param_name == "synset": continue
                    # Float params
                    if pd.isna(param_value):
                        if prop == "particleSink":
                            if param_name == "conditions": 
                                param_value = {}
                            elif param_name == "default_physical_conditions": 
                                param_value = []
                            elif param_name == "default_visual_conditions":
                                param_value = None
                            else:
                                raise ValueError("Unhandled parameter with NaN value")
                    # TODO the rest of this inconsistent hardcoding
                    try: 
                        param_value = float(param_value)
                    # TODO handle conditions params
                    # TODO handle empty string params
                    except ValueError:
                        pass


                    # TODO remove this cookable hardcode once annotations are complete
                    if prop == "cookable" and param_name == "cook_temperature" and pd.isna(param_value):
                        print(param_record["synset"])
                        syns_to_props[param_record["synset"]][prop][param_name] = 58.
                        continue

                    syns_to_props[param_record["synset"]][prop][param_name] = param_value

    with open(PARAMS_OUTFILE_FN, "w") as f:
        json.dump(syns_to_props, f, indent=4)


if __name__ == "__main__":
    create_get_save_propagated_annots_params()

