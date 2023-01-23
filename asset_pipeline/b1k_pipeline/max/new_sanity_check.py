import sys

sys.path.append(r"D:\ig_pipeline")

import collections
import json
import pathlib
import traceback

import numpy as np
import pandas as pd
import pymxs
import trimesh.transformations
from scipy.spatial.transform import Rotation

import b1k_pipeline.utils

rt = pymxs.runtime

MAX_VERTICES = 100000
WARN_VERTICES = 20000

OUTPUT_FILENAME = "sanitycheck.json"


ALLOWED_TAGS = {
    "soft",
    "glass",
    "openable",
    "openable_both_sides",
}

ALLOWED_PART_TAGS = {
    "sub_part",
    "extra_part",
}


ALLOWED_META_IS_DIMENSIONLESS = {
    'particleremover': False,
    'watersource': True,
    'waterdrain': False,
    'heatsource': True,
    'cleaningtoolarea': False,
    'slicer': False,
    'togglebutton': True,
    'attachment': True,
    'fillable': False,
}


def get_required_meta_links(category):
    return set()


def is_light_required(category):
    # TODO: Make this more robust.
    return "light" in category or "lamp" in category


def compute_shear(obj):
    # Check that object satisfies no-shear rule.
    object_transform = np.hstack(
        [np.array(obj.objecttransform), [[0], [0], [0], [1]]]
    ).T
    transform = np.hstack([np.array(obj.transform), [[0], [0], [0], [1]]]).T
    obj_to_pivot = np.linalg.inv(transform) @ object_transform
    shear = trimesh.transformations.decompose_matrix(np.linalg.inv(obj_to_pivot))[1]
    return shear


def get_approved_categories():
    # gc = gspread.service_account(filename=KEY_FILE)
    # sh = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)

    exists = []
    approved = []

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
        is_valid_type = row.type in [
            rt.Editable_Poly,
            rt.VRayLight,
            rt.VRayPhysicalCamera,
            rt.Point,
            rt.Box,
            rt.Cylinder,
            rt.Sphere,
            rt.Cone,
            rt.Physical_Camera,
            rt.FreeCamera,
        ]

        self.expect(is_valid_type, f"{row.object_name} has disallowed type {row.type}.")
        if row.object.parent is None:
            is_valid_root = row.type in [
                rt.Editable_Poly,
                rt.VRayLight,
                rt.VRayPhysicalCamera,
                rt.Physical_Camera,
                rt.FreeCamera,
            ]
            self.expect(is_valid_root, f"Root-level object {row.object_name} has disallowed type {row.type}.")
        return is_valid_type

    def is_valid_name(self, row):
        is_valid_name = (
            not pd.isna(row.object_name)
            and b1k_pipeline.utils.parse_name(row.object_name) is not None
        )
        self.expect(is_valid_name, f"{row.object_name} has bad name.")
        return is_valid_name

    def validate_object(self, row):
        # Check that the category exists on the spreadsheet
        # self.expect(
        #   row.name_category in self._existing_categories,
        #   f"Category {row.name_category} for object {row.object_name} does not exist on spreadsheet.",
        #   level="WARNING")

        # Check that the category is approved on the spreadsheet
        # self.expect(
        #   row.name_category in self._existing_categories,
        #   f"Category {row.name_category} for object {row.object_name} is not approved on spreadsheet.",
        #   level="WARNING")

        # Check that it does not have too many vertices.
        # self.expect(
        #   row.vertex_count <= MAX_VERTICES,
        #   f"{row.object_name} has too many vertices: {row.vertex_count} > {MAX_VERTICES}")

        # Warn if it has too many but acceptable vertices.
        self.expect(
            row.vertex_count <= WARN_VERTICES,
            f"{row.object_name} has too many vertices: {row.vertex_count} > {WARN_VERTICES}",
            level="WARNING",
        )

        # Check that the object does not have any modifiers
        self.expect(
            len(row.object.modifiers) == 0, f"{row.object_name} has modifiers attached."
        )

        # Check that object satisfies the scale condition.
        scale = np.array(row.object.scale)
        self.expect(
            np.allclose(scale, 1),
            f"{row.object_name} has scale that is not 1. Reset scale.",
        )

        # Check that object does not have negative object offset scale.
        object_offset_scale = np.array(row.object.objectoffsetscale)
        self.expect(
            np.all(object_offset_scale > 0),
            f"{row.object_name} has negative object offset scale. Try mirroring + resetting scale.",
        )

        # Check that object does not have a light ID.
        self.expect(
            pd.isna(row.name_light_id),
            f"Non-light object {row.object_name} should not have light ID.",
        )

        # Check that object does not have any meta type.
        self.expect(
            pd.isna(row.name_meta_info),
            f"Non-meta object {row.object_name} should not have meta info.",
        )

        # Check object name format.
        self.expect(
            len(row.name_model_id) == 6,
            f"{row.object_name} does not have 6-digit model ID.",
        )

        # Check the tags
        tags = set()
        tags_str = row.name_tag
        if tags_str:
            tags = set([x[1:] for x in tags_str.split("-") if x])
        invalid_tags = tags - (ALLOWED_TAGS | ALLOWED_PART_TAGS)
        self.expect(not invalid_tags, f"{row.object_name} has invalid tags {invalid_tags}.")

        part_tags = tags & ALLOWED_PART_TAGS
        if row.object.parent is None:
            self.expect(not part_tags, f"{row.object_name} is not under another object but contains part tags {part_tags}.")
        else:
            self.expect(part_tags, f"Part object {row.object_name} should be marked with a part tag, one of: {ALLOWED_PART_TAGS}.")

        # Validate meta stuff
        if int(row.name_instance_id) == 0:
            self.validate_meta_links(row)

    def validate_light(self, row):
        # Validate that the object has a light ID
        self.expect(
            not pd.isna(row.name_light_id),
            f"Light object {row.object_name} should have light ID.",
        )

        # Check that the type is one of plane, sphere, disk
        self.expect(
            row.object.type in (0, 2, 4),
            f"Light object {row.object_name} type should be one of SPHERE, DISC, PLANE",
        )

        # Light should not be scaled or offset
        self.expect(
            np.allclose(row.object.scale, 1),
            f"Light object {row.object_name} should not have scale.",
        )
        self.expect(
            np.allclose(row.object.objectoffsetscale, 1),
            f"Light object {row.object_name} should not have scale.",
        )
        self.expect(
            np.allclose(row.object.objectoffsetpos, 0),
            f"Light object {row.object_name} should not have object offset position. Reset pivot.",
        )
        self.expect(
            np.allclose(quat2arr(row.object.objectoffsetrot), [0, 0, 0, 1]),
            f"Light object {row.object_name} should not have object offset rotation. Reset pivot.",
        )

    def validate_link_set(self, rows):
        # The rows here should correspond to individual positions of the same link.
        base = rows.iloc[0]
        # TODO: Assert that these exist, they are instances, named correctly, etc.

    def validate_group_of_instances(self, rows):
        # Pick an object as the base instance
        # TODO: Do a better job of this.
        rows_with_id_zero = rows[rows["name_instance_id"] == "0"]
        obj_name = rows["object_name"].iloc[0]
        assert (
            len(rows_with_id_zero.index) > 0
        ), f"No instance ID 0 instance of {obj_name}."
        base = rows_with_id_zero.iloc[0]

        # Check that they have the same model ID and same category
        unique_model_ids = rows.groupby(
            ["name_category", "name_model_id"], sort=False, dropna=False
        ).ngroups
        self.expect(
            unique_model_ids == 1,
            f"All instances of {base.object_name} do not share category/model ID.",
        )

        # Check that the keys start from 0 and are contiguous.
        unique_instance_ids = rows.groupby(
            ["name_category", "name_model_id", "name_instance_id"],
            sort=False,
            dropna=False,
        ).ngroups
        existing_instance_ids = set(rows["name_instance_id"])
        expected_instance_ids = set(str(x) for x in range(unique_instance_ids))
        self.expect(
            existing_instance_ids == expected_instance_ids,
            f"All instances of {base.object_name} do not have contiguous instance IDs. Missing: {sorted(expected_instance_ids - existing_instance_ids)}",
        )

        # Check that they all have the same object offset rotation and pos/scale and shear.
        desired_offset_pos = np.array(base.object.objectOffsetPos) / np.array(
            base.object.objectOffsetScale
        )
        desired_offset_rot_inv = Rotation.from_quat(
            quat2arr(base.object.objectOffsetRot)
        ).inv()
        desired_shear = compute_shear(base.object)
        for _, row in rows.iterrows():
            this_offset_pos = np.array(row.object.objectOffsetPos) / np.array(
                row.object.objectOffsetScale
            )
            pos_diff = this_offset_pos - desired_offset_pos
            self.expect(
                np.allclose(pos_diff, 0, atol=5e-2),
                f"{row.object_name} has different pivot offset position (by {pos_diff}). Match pivots on each instance.",
            )

            this_offset_rot = Rotation.from_quat(quat2arr(row.object.objectOffsetRot))
            rot_diff = (this_offset_rot * desired_offset_rot_inv).magnitude()
            self.expect(
                np.allclose(rot_diff, 0, atol=1e-3),
                f"{row.object_name} has different pivot offset rotation (by {rot_diff}). Match pivots on each instance.",
            )

            this_shear = compute_shear(row.object)
            self.expect(
                np.allclose(this_shear, desired_shear, atol=1e-3),
                f"{row.object_name} has different shear. Match scaling axes on each instance.",
            )

            self.expect(
                row.object.material == base.object.material,
                f"{row.object_name} has different material. Match materials on each instance.",
            )

    def validate_meta_links(self, row):
        # Don't worry about bad objects or upper links.
        if row.name_joint_side == "upper":
            return

        # Get the children using the parent rows
        children = row.object.children

        # Assert that all of the children are valid meta-links / lights
        found_ml_types = set()
        found_ml_subids_for_id = collections.defaultdict(lambda: collections.defaultdict(list))
        for child in children:
            # If it's an EditablePoly, then it will be processed as a root object too. Safe to skip.
            if rt.classOf(child) == rt.Editable_Poly:
                continue

            # Otherwise, we can validate the individual meta link
            match = b1k_pipeline.utils.parse_name(child.name)
            if match is None:
                self.expect(False, f"{child.name} is an invalid meta link under {row.object_name}.")
                continue
            self.expect(match.group("mesh_basename") == row.name_mesh_basename, f"Meta link {child.name} base name doesnt match parent {row.object_name}")

            # Keep track of the meta links we have
            self.expect(match.group("meta_type"), f"Meta object {child.name} should have a meta link type in its name.")
            meta_link_type = match.group("meta_type")
            if meta_link_type not in ALLOWED_META_IS_DIMENSIONLESS:
                self.expect(False, f"Meta link type {meta_link_type} not in list of allowed meta link types: {ALLOWED_META_IS_DIMENSIONLESS.keys()}")
                continue
            
            found_ml_types.add(meta_link_type)

            # Keep track of subids for this meta ID
            meta_id = "0"
            if match.group("meta_id"):
                meta_id = match.group("meta_id")
            meta_subid = "0"
            if match.group("meta_subid"):
                meta_subid = match.group("meta_subid")
            found_ml_subids_for_id[meta_link_type][meta_id].append(meta_subid)

            if ALLOWED_META_IS_DIMENSIONLESS[meta_link_type]:
                self.expect(rt.classOf(child) == rt.Point, f"Dimensionless {meta_link_type} meta link {child.name} should be of Point instead of {rt.classOf(child)}")
            else:
                volumetric_allowed_types = {
                    rt.Box,
                    rt.Cylinder,
                    rt.Sphere,
                    rt.Cone,
                }
                self.expect(rt.classOf(child) in volumetric_allowed_types, f"Volumetric {meta_link_type} meta link {child.name} should be one of {volumetric_allowed_types} instead of {rt.classOf(child)}")

                if rt.classOf(child) == rt.Cone:
                    # Cones should have radius1 as 0 and radius2 nonzero
                    self.expect(np.isclose(child.radius1, 0), f"Cone {child.name} radius1 should be zero.")
                    self.expect(not np.isclose(child.radius2, 0), f"Cone {child.name} radius2 should be nonzero.")

        # Check that the meta links match what's needed
        required_meta_types = get_required_meta_links(row.name_category)
        missing_meta_types = required_meta_types - found_ml_types
        self.expect(not missing_meta_types, f"Expected meta types for {row.object_name} are missing: {missing_meta_types}")

        # Check that meta subids are correct:
        for meta_type, meta_ids_to_subids in found_ml_subids_for_id.items():
            for meta_id, meta_subids in meta_ids_to_subids.items():
                expected_subids = {str(x) for x in range(len(meta_subids))}
                self.expect(set(meta_subids) == expected_subids, f"{row.name_mesh_basename} meta type {meta_type} ID {meta_id} has non-continuous subids {sorted(meta_subids)}")

    def run(self):
        self.reset()

        self.expect(rt.units.systemScale == 1, "System scale not set to 1mm.")

        self.expect(
            rt.units.systemType == rt.Name("millimeters"),
            "System scale not set to 1mm.",
        )

        # First get all of the objects
        df = pd.DataFrame(data={"object": list(rt.objects)})

        # Add some helpful data
        df["object_name"] = df["object"].map(lambda x: x.name)
        df["base_object"] = df["object"].map(lambda x: x.baseObject)
        df["vertex_count"] = df["object"].map(lambda x: len(x.vertices))
        df["type"] = df["object"].map(lambda x: rt.classOf(x))

        # Complain about and remove objects that are the wrong type.
        good_type = df.apply(self.is_valid_object_type, axis="columns")
        df = df[good_type]

        # Additionally filter out the cameras
        df = df[df["type"].isin([rt.Editable_Poly, rt.VRayLight])]

        # Complain about and remove objects that have unmatching name.
        good_name = df.apply(self.is_valid_name, axis="columns")
        df = df[good_name]

        if df.size == 0:
            return self.errors

        # Check that the object names are unique.
        duplicate_named_objs = df[df.duplicated(subset=["object_name"], keep="first")]
        duplicate_named_objs.apply(
            lambda row: self.expect(False, f"{row.object_name} is not unique.")
        )

        # Unwrap the name into its columns
        col_to_name = {
            v - 1: "name_" + k
            for k, v in b1k_pipeline.utils.NAME_PATTERN.groupindex.items()
        }
        applied_df = df.apply(
            lambda row: list(b1k_pipeline.utils.parse_name(row.object_name).groups()),
            axis="columns",
            result_type="expand",
        )
        applied_df.rename(columns=col_to_name, inplace=True)
        df = pd.concat([df, applied_df], axis="columns")
        df["name_link_name"] = df["name_link_name"].replace(
            "", "base_link"
        )  # no-link-name objects are matched with the base link.
        columns = set(df.columns)

        # Run the single-object validation checks.
        objs = df[(df["type"] == rt.Editable_Poly) & df["name_bad"].isnull()]
        objs.apply(self.validate_object, axis="columns")

        # Check that instance name-based grouping is equal to instance-based grouping.
        groups_by_base_object = objs.groupby(["base_object"], sort=False, dropna=False)
        groups_by_base_object.apply(
            lambda group: self.expect(
                group.groupby(
                    ["name_category", "name_model_id", "name_link_name"],
                    sort=False,
                    dropna=False,
                ).ngroups
                == 1,
                f"{group.iloc[0].object_name} has instances that are not numbered as instances. If this is a link, note that different links should not be instances of each other.",
            )
        )
        groups_by_instance_mark = objs.groupby(
            ["name_category", "name_model_id", "name_link_name"],
            sort=False,
            dropna=False,
        )
        groups_by_instance_mark.apply(
            lambda group: self.expect(
                group.groupby(["base_object"], sort=False, dropna=False).ngroups == 1,
                f"{group.iloc[0].object_name} has objects that are not actually instances marked as instances.",
            )
        )

        # Let's use the base-object grouping for the instance group validation:
        groups_by_base_object.apply(self.validate_group_of_instances)

        lights = df[df["type"] == rt.VRayLight]
        lights.apply(self.validate_light, axis="columns")

        # TODO: Check that each light object has a valid base.
        # TODO: Check that each light object is numbered consecutively.
        # TODO: Check that light-named objects have lights on them.

        return self.errors


def sanity_check_safe(batch=False):
    success = False
    errors = []
    warnings = []

    try:
        results = SanityCheck().run()
        errors = results["ERROR"]
        warnings = results["WARNING"]
        success = len(errors) == 0
    except Exception as e:
        t = traceback.format_exc()
        errors.append("Exception occurred:" + t)

    # Print results in interactive mode.
    if not batch:
        if success:
            print("Sanity check complete - no errors found!")
        else:
            print("Errors found:")
            print("\n".join(errors))

        if warnings:
            print("Warnings:")
            print("\n".join(warnings))

    if batch:
        output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / OUTPUT_FILENAME, "w") as f:
            json.dump(
                {"success": success, "errors": errors, "warnings": warnings},
                f,
                indent=4,
            )


if __name__ == "__main__":
    opts = rt.maxops.mxsCmdLineArgs

    batch = opts[rt.name("batch")] == "true"
    sanity_check_safe(batch=batch)
