import json
import pathlib
from pymxs import runtime as rt
from tqdm import tqdm

import sys
sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.max.prebake_textures import hash_object

OUTPUT_FILENAME = "file_manifest.json"

def hash_material(root_mat):
    """
    Get the hash of a material by recursively hashing its sub-materials.
    This is a very blind hash - it will just identify if ANYTHING in the hierarchy
    of materials has changed.
    """

    # Implement a try/except for getattr to avoid crashing if the attribute doesn't exist.
    # This seems to avoid issues around bitmaptextures not having bitmaps when the file is not
    # found.
    def _try_to_getattr(_obj, attr):
        try:
            return getattr(_obj, attr)
        except:
            return None

    hash_elems = []
    def _recursively_hash_materials_and_textures(mtl):
        # Do the actual hashing
        hash_elems.append(hash(tuple([(k, _try_to_getattr(mtl, k)) for k in dir(mtl)])))

        # We can check for submtls for material instances
        if rt.superClassOf(mtl) == rt.Material:
            for i in range(rt.getNumSubMtls(mtl)):
                sub_mtl = rt.getSubMtl(mtl, i + 1)
                if sub_mtl is not None:
                    _recursively_hash_materials_and_textures(sub_mtl)

        # We can check for subtexmaps for texture maps and materials
        if rt.superClassOf(mtl) == rt.textureMap or rt.superClassOf(mtl) == rt.Material:
            for i in range(rt.getNumSubTexmaps(mtl)):
                sub_texmap = rt.getSubTexmap(mtl, i + 1)
                if sub_texmap is not None:
                    _recursively_hash_materials_and_textures(sub_texmap)

    _recursively_hash_materials_and_textures(root_mat)
    return hash(tuple(hash_elems))

def main():
    # Go through all the objects and store their information.
    file_manifest = []
    for obj in tqdm(list(rt.objects)):
        obj_id = obj.inode.handle
        obj_name = obj.name
        obj_parent = obj.parent.name if obj.parent else None
        obj_layer = obj.layer.name
        obj_class = str(rt.classOf(obj))
        if rt.classOf(obj) == rt.Editable_Poly:
            obj_hash = hash_object(obj)
        else:
            obj_hash = hash(tuple([(k, getattr(obj, k)) for k in dir(obj)]))
        obj_mtl_hash = hash_material(obj.material) if obj.material else None
        obj_transform = str(obj.transform)
        obj_objecttransform = str(obj.objectTransform)
        file_manifest.append({
            "id": obj_id,
            "name": obj_name,
            "parent": obj_parent,
            "layer": obj_layer,
            "class": obj_class,
            "hash": obj_hash,
            "mtl_hash": obj_mtl_hash,
            "transform": obj_transform,
            "objecttransform": obj_objecttransform,
        })

    # Sort the list by id
    file_manifest.sort(key=lambda x: x["id"])

    # Dump into a JSON file
    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    with open(filename, "w") as f:
        json.dump(file_manifest, f, indent=4)

if __name__ == "__main__":
    main()
