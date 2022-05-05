import collections
import json
import re
import pandas as pd
import numpy as np
import trimesh.transformations
import os

import gspread

try:
  import pymxs
  rt = pymxs.runtime
except:
  print("Could not import pymxs. Are you in 3ds Max?")

MAX_VERTICES = 100000
WARN_VERTICES = 20000
PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?$")

SPREADSHEET_ID = "1JJob97Ovsv9HP1Xrs_LYPlTnJaumR2eMELImGykD22A"
WORKSHEET_NAME = "Object Category B1K"
KEY_FILE = os.path.join(os.path.dirname(__file__), "../keys/b1k-dataset-6966129845c0.json")


def compute_shear(obj):
  # Check that object satisfies no-shear rule.
  object_transform = np.hstack([np.array(obj.objecttransform), [[0], [0], [0], [1]]]).T
  transform = np.hstack([np.array(obj.transform), [[0], [0], [0], [1]]]).T
  obj_to_pivot = np.linalg.inv(transform) @ object_transform
  shear = trimesh.transformations.decompose_matrix(np.linalg.inv(obj_to_pivot))[1]
  return shear


def get_approved_categories():
  gc = gspread.service_account(filename=KEY_FILE)
  sh = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)

  exists = []
  approved = []
  for row in sh.get_values()[1:]:
    name = row[0]
    if not name:
      continue

    exists.append(name)
    if row[3]:
      approved.append(name)

  return exists, approved

def quat2arr(q):
  return np.array([q.x, q.y, q.z, q.w])

class SanityCheck:
  def __init__(self):
    self.reset()
    self._existing_categories, self._approved_categories = get_approved_categories()

  def expect(self, condition, message, level="ERROR"):
    if not condition:
      self.errors[level].append(message)

  def reset(self):
    self.errors = collections.defaultdict(list)

  def is_valid_object_type(self, row):
    is_valid_type = row.type in [rt.Editable_Poly, rt.VRayLight, rt.VRayPhysicalCamera]
    self.expect(is_valid_type, f"{row.object_name} has disallowed type {row.type}.")
    return is_valid_type

  def is_valid_name(self, row):
    is_valid_name = not pd.isna(row.object_name) and PATTERN.fullmatch(row.object_name) is not None
    self.expect(is_valid_name, f"{row.object_name} has bad name.")
    return is_valid_name

  def validate_object(self, row):
    # Check that the category exists on the spreadsheet
    self.expect(
      row.name_category in self._existing_categories,
      f"Category {row.name_category} for object {row.object_name} does not exist on spreadsheet.")

    # Check that the category is approved on the spreadsheet
    self.expect(
      row.name_category in self._existing_categories,
      f"Category {row.name_category} for object {row.object_name} is not approved on spreadsheet.",
      level="WARNING")

    # Check that it does not have too many vertices.
    self.expect(
      row.vertex_count <= MAX_VERTICES,
      f"{row.object_name} has too many vertices: {row.vertex_count} > {MAX_VERTICES}")

    # Warn if it has too many but acceptable vertices.
    self.expect(
      row.vertex_count <= WARN_VERTICES or row.vertex_count > MAX_VERTICES,
      f"{row.object_name} has too many vertices: {row.vertex_count} > {MAX_VERTICES}",
      level="WARNING")

    # Check that the object does not have any modifiers
    self.expect(len(row.object.modifiers) == 0, f"{row.object_name} has modifiers attached.")

    # Check that object satisfies the scale condition.
    scale = np.array(row.object.scale)
    self.expect(np.allclose(scale, 1), f"{row.object_name} has scale that is not 1. Reset scale.")

    # Check that object does not have negative object offset scale.
    object_offset_scale = np.array(row.object.objectoffsetscale)
    self.expect(
      np.all(object_offset_scale > 0),
      f"{row.object_name} has negative object offset scale. Try mirroring + resetting scale.")

    # Check that object does not have a light ID.
    self.expect(pd.isna(row.name_light_id), f"Non-light object {row.object_name} should not have light ID.")

    # Check object name format.
    self.expect(len(row.name_model_id) == 6, f"{row.object_name} does not have 6-digit model ID.")

  def validate_light(self, row):
    # Validate that the object has a light ID
    self.expect(not pd.isna(row.name_light_id), f"Light object {row.object_name} should have light ID.")

    # Check that the type is one of plane, sphere, disk
    self.expect(row.object.type in (0, 2, 4), f"Light object {row.object_name} type should be one of SPHERE, DISC, PLANE")

    # Light should not be scaled or offset
    self.expect(np.allclose(row.object.scale, 1), f"Light object {row.object_name} should not have scale.")
    self.expect(np.allclose(row.object.objectoffsetscale, 1), f"Light object {row.object_name} should not have scale.")
    self.expect(np.allclose(row.object.objectoffsetpos, 0), f"Light object {row.object_name} should not have object offset position. Reset pivot.")
    self.expect(np.allclose(quat2arr(row.object.objectoffsetrot), [0, 0, 0, 1]), f"Light object {row.object_name} should not have object offset rotation. Reset pivot.")

  def validate_link_set(self, rows):
    # The rows here should correspond to individual positions of the same link.
    base = rows.iloc[0]
    # TODO: Assert that these exist, they are instances, named correctly, etc.


  def validate_group_of_instances(self, rows):
    # Pick an object as the base instance
    # TODO: Do a better job of this.
    base = rows.iloc[0]

    # Check that they have the same model ID and same category
    unique_model_ids = rows.groupby(["name_category", "name_model_id"], sort=False, dropna=False).ngroups
    self.expect(unique_model_ids == 1, f"All instances of {base.object_name} do not share category/model ID.")

    # Check that the keys start from 0 and are contiguous.
    unique_instance_ids = rows.groupby(["name_category", "name_model_id", "name_instance_id"], sort=False, dropna=False).ngroups
    self.expect(set(rows["name_instance_id"]) == set(str(x) for x in range(unique_instance_ids)), f"All instances of {base.object_name} do not have contiguous instance IDs.")

    # Check that they all have the same object offset rotation and pos/scale and shear.
    desired_offset_pos = np.array(base.object.objectOffsetPos) / np.array(base.object.objectOffsetScale)
    desired_offset_rot = quat2arr(base.object.objectOffsetRot)
    desired_shear = compute_shear(base.object)
    for _, row in rows.iterrows():
        this_offset_pos = np.array(row.object.objectOffsetPos) / np.array(row.object.objectOffsetScale)
        self.expect(
          np.allclose(this_offset_pos, desired_offset_pos),
          f"{row.object_name} has different pivot offset rotation. Match pivots on each instance.")

        this_offset_rot = quat2arr(row.object.objectOffsetRot)
        self.expect(
          np.allclose(this_offset_rot, desired_offset_rot),
          f"{row.object_name} has different pivot offset rotation. Match pivots on each instance.")
        
        this_shear = compute_shear(row.object)
        self.expect(
          np.allclose(this_shear, desired_shear, atol=1e-5),
          f"{row.object_name} has different shear. Match scaling axes on each instance."
        )

        self.expect(
          row.object.material == base.object.material,
          f"{row.object_name} has different material. Match materials on each instance.")

  def run(self):
    self.reset()

    self.expect(
      rt.units.systemScale == 1,
      "System scale not set to 1mm.")

    self.expect(
      rt.units.systemType == rt.Name("millimeters"),
      "System scale not set to 1mm."
    )

    # First get all of the objects
    df = pd.DataFrame(data={"object": list(rt.objects)})

    # Add some helpful data
    df["object_name"] = df["object"].map(lambda x: x.name)
    df["base_object"] = df["object"].map(lambda x: x.baseObject)
    df["vertex_count"] = df["object"].map(lambda x: len(x.vertices))
    df["type"] = df["object"].map(lambda x: rt.classOf(x))

    # Complain about and remove objects that are the wrong type or unmatching name.
    good_type = df.apply(self.is_valid_object_type, axis="columns")
    good_name = df.apply(self.is_valid_name, axis="columns")
    df = df[good_type & good_name]

    # Additionally filter out the cameras
    df = df[df["type"].isin([rt.Editable_Poly, rt.VRayLight])]

    if df.size == 0:
      return self.errors

    # Check that the object names are unique.
    duplicate_named_objs = df[df.duplicated(subset=["object_name"], keep="first")]
    duplicate_named_objs.apply(lambda row: self.expect(False, f"{row.object_name} is not unique."))     

    # Unwrap the name into its columns
    col_to_name = {v-1: "name_" + k for k, v in PATTERN.groupindex.items()}
    applied_df = df.apply(lambda row: list(PATTERN.fullmatch(row.object_name).groups()), axis='columns', result_type='expand')
    applied_df.rename(columns=col_to_name, inplace=True)
    df = pd.concat([df, applied_df], axis='columns')
    df["name_link_name"] = df["name_link_name"].replace("", "base_link")  # no-link-name objects are matched with the base link.
    columns = set(df.columns)

    # Run the single-object validation checks.
    objs = df[df["type"] == rt.Editable_Poly]
    objs.apply(self.validate_object, axis="columns")

    # Check that instance name-based grouping is equal to instance-based grouping.
    groups_by_base_object = objs.groupby(["base_object"], sort=False, dropna=False)
    groups_by_base_object.apply(
      lambda group: self.expect(
        group.groupby(["name_category", "name_model_id", "name_link_name"], sort=False, dropna=False).ngroups == 1,
        f"{group.iloc[0].object_name} has instances that are not numbered as instances. If this is a link, note that different links should not be instances of each other.",
      ))
    groups_by_instance_mark = objs.groupby(["name_category", "name_model_id", "name_link_name"], sort=False, dropna=False)
    groups_by_instance_mark.apply(
      lambda group: self.expect(
        group.groupby(["base_object"], sort=False, dropna=False).ngroups == 1,
        f"{group.iloc[0].object_name} has objects that are not actually instances marked as instances.",
      ))

    # Let's use the base-object grouping for the instance group validation:
    groups_by_base_object.apply(self.validate_group_of_instances)

    lights = df[df["type"] == rt.VRayLight]
    lights.apply(self.validate_light, axis='columns')
    
    # TODO: Check that each light object has a valid base.
    # TODO: Check that each light object is numbered consecutively.
    # TODO: Check that light-named objects have lights on them.

    return self.errors

def create_macroscript(
        _func, category="", name="", tool_tip="", button_text="", *args):
    """Creates a macroscript"""

    try:
        # gets the qualified name for bound methods
        # ex: data_types.general_types.GMesh.center_pivot
        func_name = "{0}.{1}.{2}".format(
            _func.__module__, args[0].__class__.__name__, _func.__name__)
    except (IndexError, AttributeError):
        # gets the qualified name for unbound methods
        # ex: data_types.general_types.get_selection
        func_name = "{0}.{1}".format(
            _func.__module__, _func.__name__)

    script = """
    (
        python.Execute "import {}"
        python.Execute "{}()"
    )
    """.format(_func.__module__, func_name)
    rt.macros.new(category, name, tool_tip, button_text, script)

def sanity_check_safe():
  errors = SanityCheck().run()

  if errors["ERROR"]:
    print("Errors found:")
    print("\n".join(errors["ERROR"]))
  else:
    print("Sanity check complete - no errors found!")

  if errors["WARNING"]:
    print("Warnings:")
    print("\n".join(errors["WARNING"]))

def sanity_check_batch(output_path):
  success = False
  errors = []
  warnings = []

  try:
    results = SanityCheck().run()
    errors = results["ERROR"]
    warnings = results["WARNING"]
    success = len(errors) == 0
  except Exception as e:
    traceback = traceback.format_exc()
    errors.append("Exception occurred:" + traceback)

  with open(output_path, "w") as f:
    json.dump({"success": success, "errors": errors, "warnings": warnings}, f, indent=4)


if __name__ == "__main__":
  opts = rt.maxops.mxsCmdLineArgs

  if opts[rt.name('batch')] == "true":
    sanity_check_batch(opts[rt.name('output_path')])
  else:
    create_macroscript(sanity_check_safe, category="SVL-Tools", name="Sanity Check", button_text="Sanity Check")