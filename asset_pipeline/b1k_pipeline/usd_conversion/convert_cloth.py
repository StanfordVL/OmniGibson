import os
import shutil
import numpy as np
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.kit.commands import execute
from omnigibson.macros import gm

def postprocess_cloth(usd_file):
    # Copy to new path
    cloth_usd_file = usd_file.replace(".usd", "_cloth.usd")
    shutil.copy2(usd_file, cloth_usd_file)

    # Open the new file
    assert open_stage(cloth_usd_file)
    stage = get_current_stage()
    prim = stage.GetDefaultPrim()

    # Collapse the typical structure to a single visual mesh
    links = [child for child in prim.GetChildren() if child.GetTypeName() == "Xform"]
    assert len(links) == 1, f"[{cloth_usd_file}] has more than one link"
    link = links[0]

    tmp_base_link_path = prim.GetPrimPath().AppendPath("base_link_backup")
    execute("MovePrim", path_from=link.GetPrimPath(), path_to=tmp_base_link_path)

    link = get_prim_at_path(tmp_base_link_path)

    visual_mesh_path = tmp_base_link_path.AppendPath("visuals")
    assert is_prim_path_valid(visual_mesh_path), f"WARNING: visual mesh path [{visual_mesh_path}] does not exist"

    visual_mesh = get_prim_at_path(visual_mesh_path)
    assert visual_mesh.GetTypeName() == "Mesh", \
        f"WARNING: visual mesh path [{visual_mesh_path}] does not have type Mesh " \
        f"(likely because there are multiple visual meshes)."

    execute("MovePrim", path_from=visual_mesh.GetPrimPath(), path_to=prim.GetPrimPath().AppendPath("base_link"))

    stage.RemovePrim(tmp_base_link_path)
    stage.Save()

    print(f"Done: {cloth_usd_file}")