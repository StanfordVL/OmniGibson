import os
import shutil
from omnigibson import app, og_dataset_path
from refactor_scripts.import_scene_template import import_models_template_from_scene

old_og_dataset_path = "/cvgl2/u/chengshu/iGibson3/omnigibson/data/ig_dataset_new"
if __name__ == "__main__":
    # TODO: these should be re-generated
    for scene_id in os.listdir(os.path.join(og_dataset_path, "scenes")):
        scene_dir = os.path.join(os.path.join(og_dataset_path, "scenes"), scene_id)
        urdf = f"{scene_dir}/urdf/{scene_id}_best.urdf"
        usd_out = f"{scene_dir}/usd/{scene_id}_best_template.usd"
        import_models_template_from_scene(urdf=urdf, usd_out=usd_out)

        old_scene_dir = os.path.join(old_og_dataset_path, "scenes", scene_id)
        if os.path.isdir(os.path.join(scene_dir, "layout")):
            shutil.rmtree(os.path.join(scene_dir, "layout"))
        shutil.copytree(os.path.join(old_scene_dir, "layout"), os.path.join(scene_dir, "layout"))

    for filename in ["room_categories.txt", "non_sampleable_categories.txt", "avg_category_specs.json"]:
        shutil.copy(f"{old_og_dataset_path}/metadata/{filename}",
                    f"{og_dataset_path}/metadata/{filename}")

    app.close()
