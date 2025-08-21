def _load_scene_from_urdf(urdf):
    """
    Loads a scene from a URDF file.

    Args:
        urdf (str): Path to the URDF file.

    Raises:
        ValueError: If an object fails to load.

    This function performs the following steps:
    1. Extracts object configuration information from the URDF file.
    2. Creates a new scene without a floor plane and imports it into the simulator.
    3. Iterates over the objects' information and attempts to load each object into the scene.
       - If the USD file for an object does not exist, it prints a message and skips the object.
       - If an object fails to load, it raises a ValueError with the object's name.
    4. Sets the bounding box center position and orientation for each loaded object.
    5. Takes a simulation step to finalize the scene setup.
    """
    # First, grab object info from the urdf
    objs_info = _get_objects_config_from_scene_urdf(urdf=urdf)

    # Load all the objects manually into a scene
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)

    for obj_name, obj_info in objs_info.items():
        try:
            if not os.path.exists(
                DatasetObject.get_usd_path(obj_info["cfg"]["category"], obj_info["cfg"]["model"]).replace(
                    ".usd", ".encrypted.usd"
                )
            ):
                log.warning("Missing object", obj_name)
                continue
            obj = DatasetObject(
                name=obj_name,
                **obj_info["cfg"],
            )
            scene.add_object(obj)
            obj.set_bbox_center_position_orientation(position=obj_info["bbox_pos"], orientation=obj_info["bbox_quat"])
        except Exception as e:
            raise ValueError(f"Failed to load object {obj_name}") from e

    # Take a sim step
    og.sim.step()


def convert_scene_urdf_to_json(urdf, json_path):
    """
    Converts a scene from a URDF file to a JSON file.

    This function loads the scene described by the URDF file into the OmniGibson simulator,
    plays the simulation, and saves the scene to a JSON file. After saving, it removes the
    "init_info" from the JSON file and saves it again.

    Args:
        urdf (str): The file path to the URDF file describing the scene.
        json_path (str): The file path where the JSON file will be saved.
    """
    # First, load the requested objects from the URDF into OG
    _load_scene_from_urdf(urdf=urdf)

    # Play the simulator, then save
    og.sim.play()
    Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
    og.sim.save(json_paths=[json_path])

    # Load the json, remove the init_info because we don't need it, then save it again
    with open(json_path, "r") as f:
        scene_info = json.load(f)

    scene_info.pop("init_info")

    with open(json_path, "w+") as f:
        json.dump(scene_info, f, cls=_TorchEncoder, indent=4)
