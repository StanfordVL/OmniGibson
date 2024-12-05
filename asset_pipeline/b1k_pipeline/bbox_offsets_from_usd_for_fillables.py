from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import tempfile
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom, Gf
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import fs.path
from fs.zipfs import ZipFS

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
KEY_PATH = pathlib.Path(r"C:\Users\cgokmen\research\OmniGibson\omnigibson\data\omnigibson.key")
PIPELINE_ROOT = pathlib.Path(r"D:\ig_pipeline")
USE_ATTRIBUTE = False


def get_orientation_edits():
    orientation_edits = {}
    zip_path = PIPELINE_ROOT / "metadata" / "orientation_edits.zip"
    with ZipFS(zip_path) as orientation_zip_fs:
        for item in orientation_zip_fs.glob("recorded_orientation/*/*.json"):
            model = fs.path.splitext(fs.path.basename(item.path))[0]
            orientation = json.loads(orientation_zip_fs.readtext(item.path))[0]
            if np.allclose(orientation, [0, 0, 0, 1], atol=1e-3):
                continue
            orientation_edits[model] = orientation

    return orientation_edits


def _set_xform_properties(prim, pos, quat):
    properties_to_remove = [
        "xformOp:rotateX",
        "xformOp:rotateXZY",
        "xformOp:rotateY",
        "xformOp:rotateYXZ",
        "xformOp:rotateYZX",
        "xformOp:rotateZ",
        "xformOp:rotateZYX",
        "xformOp:rotateZXY",
        "xformOp:rotateXYZ",
        "xformOp:transform",
    ]
    prop_names = prim.GetPropertyNames()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # TODO: wont be able to delete props for non root links on articulated objects
    for prop_name in prop_names:
        if prop_name in properties_to_remove:
            prim.RemoveProperty(prop_name)
    if "xformOp:scale" not in prop_names:
        xform_op_scale = xformable.AddXformOp(
            UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, ""
        )
        xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
    else:
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if "xformOp:translate" not in prop_names:
        xform_op_translate = xformable.AddXformOp(
            UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        xform_op_rot = xformable.AddXformOp(
            UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

    position = Gf.Vec3d(*pos.tolist())
    xform_op_translate.Set(position)

    orientation = quat[[3, 0, 1, 2]].tolist()
    if xform_op_rot.GetTypeName() == "quatf":
        rotq = Gf.Quatf(*orientation)
    else:
        rotq = Gf.Quatd(*orientation)
    xform_op_rot.Set(rotq)


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://openusd.org/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns: 
        A range (i.e. bounding box), see more at: https://openusd.org/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return np.array(bound_range.GetMin()), np.array(bound_range.GetMax())


def get_bounding_box_from_usd(input_usd, rotmat, tempdir):
    encrypted_filename = input_usd
    fd, decrypted_filename = tempfile.mkstemp(suffix=".usd", dir=tempdir)
    os.close(fd)
    decrypt_file(encrypted_filename, decrypted_filename)
    stage = Usd.Stage.Open(str(decrypted_filename))
    prim = stage.GetDefaultPrim()

    if USE_ATTRIBUTE:
      base_link_size = np.array(prim.GetAttribute("ig:nativeBB").Get()) * 1000
      base_link_offset = np.array(prim.GetAttribute("ig:offsetBaseLink").Get()) * 1000

      # THIS IS INACCURATE FOR NON-AA ROTATIONS!
      return (rotmat @ base_link_offset).tolist(), (rotmat @ base_link_size).tolist()
    
    else:
      # Rotate the object by the rotmat and get the bounding box
      _set_xform_properties(prim, np.zeros(3), R.from_matrix(rotmat).as_quat())
      bb_min, bb_max = compute_bbox(prim)
      center = ((bb_min + bb_max) / 2) * 1000
      size = (bb_max - bb_min) * 1000
      return center.tolist(), size.tolist()

def main():
    input_usds = list(FILLABLE_DIR.glob("objects/*/*/usd/*.usd"))
    input_usds.sort(key=lambda x: x.parts[-3])
    print(len(input_usds))

    orientation_edits = get_orientation_edits()

    # Scale up
    futures = {}
    with tempfile.TemporaryDirectory() as tempdir:
      with ProcessPoolExecutor() as executor:
          for input_usd in tqdm(input_usds, desc="Queueing up jobs"):
              mdl = input_usd.parts[-3]
              rot = R.identity() if mdl not in orientation_edits else R.from_quat(orientation_edits[mdl]).inv()
              euler = rot.as_euler("xyz")
              if not np.allclose(euler % (np.pi / 2), 0, atol=np.deg2rad(5)):
                  print("Non-axis aligned rotation detected for", mdl, np.rad2deg(euler).tolist())
              future = executor.submit(get_bounding_box_from_usd, input_usd, rot.as_matrix(), tempdir)
              futures[future] = input_usd

          # Gather the results (with a tqdm progress bar)
          results = {}
          for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
              input_usd = futures[future]
              results[input_usd.parts[-3]] = future.result()

    with open(FILLABLE_DIR / "fillable_bboxes.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()