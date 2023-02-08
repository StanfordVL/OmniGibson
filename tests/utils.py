import omnigibson as og

from omnigibson.macros import gm

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True
gm.USE_GPU_DYNAMICS = True

def og_test(func):
    def wrapper():
        assert_test_scene()
        state = og.sim.dump_state()
        func()
        og.sim.load_state(state)
    return wrapper

num_objs = 0

def get_obj_cfg(name, category, model):
    global num_objs
    num_objs += 1
    return {
        "type": "DatasetObject",
        "fit_avg_dim_volume": True,
        "name": name,
        "category": category,
        "model": model,
        "position": [100, 100, num_objs * 5],       
    }

def assert_test_scene():
    if og.sim.scene is None:
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                get_obj_cfg("breakfast_table", "breakfast_table", "skczfi"),
                get_obj_cfg("bottom_cabinet", "bottom_cabinet", "immwzb"),
                get_obj_cfg("dishtowel", "dishtowel", "Tag_Dishtowel_Basket_Weave_Red"),
                get_obj_cfg("bowl", "bowl", "ajzltc"),
            ],
        }

        # Create the environment
        env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)