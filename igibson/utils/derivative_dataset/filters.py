import collections
import itertools
import random

import numpy as np
from scipy.spatial.transform import Rotation

# from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, SemanticClass
# from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID

STRUCTURE_CLASSES = ["walls", "ceilings", "floors"]


def too_close_filter(min_dist=0, max_dist=float("inf"), max_allowed_fraction_outside_threshold=0):
    def filter_fn(env, cam, objs_of_interest):
        img = cam.get_obs()["depth_linear"]
        depth_img = np.linalg.norm(img, axis=-1)
        outside_range_pixels = np.count_nonzero(np.logical_or(depth_img < min_dist, depth_img > max_dist))
        return outside_range_pixels / len(depth_img.flatten()) <= max_allowed_fraction_outside_threshold

    return filter_fn


def too_much_structure(max_allowed_fraction_of_structure):
    def filter_fn(env, cam, objs_of_interest):
        # seg = env.simulator.renderer.render(modes=("seg"))[0][:, :, 0]
        seg, seg_info = cam.get_obs()["seg_instance"]
        # seg_int = np.round(seg * MAX_CLASS_COUNT).astype(int).flatten()
        seg_int = seg.flatten()
        CLASS_NAME_TO_CLASS_ID = {seg_info[i][3]: seg_info[i][0] for i in range(len(seg_info))}
        pixels_of_wall = np.count_nonzero(np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES]))
        return pixels_of_wall / len(seg_int) < max_allowed_fraction_of_structure

    return filter_fn


def too_much_of_same_object_in_fov_filter(threshold):
    def filter_fn(env, cam, objs_of_interest):
        # seg, ins_seg = env.simulator.renderer.render(modes=("seg", "ins_seg"))
        #
        # # Get body ID per pixel
        # ins_seg = np.round(ins_seg[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        # body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
        #
        # # Use category to remove walls, floors, ceilings
        # seg_int = np.round(seg[:, :, 0] * MAX_CLASS_COUNT).astype(int)
        seg, seg_info = cam.get_obs()["seg_instance"]
        seg_int = seg.flatten()

        CLASS_NAME_TO_CLASS_ID = {seg_info[i][3]: seg_info[i][0] for i in range(len(seg_info))}
        pixels_of_wall = np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES])

        relevant_body_ids = body_ids[np.logical_not(pixels_of_wall)]

        return max(np.bincount(relevant_body_ids)) / len(body_ids.flatten()) < threshold

    return filter_fn


def no_relevant_object_in_fov_filter(target_state, min_bbox_vertices_in_fov=4):
    def filter_fn(env, cam, objs_of_interest):
        camera_params = cam.get_obs()["camera"]
        view_proj_matrix = camera_params["view_projection_matrix"]
        resolution = camera_params["resolution"]

        bbox_annotations = cam.get_obs()["bbox_3d"]
        bboxes = {}
        for bbox_annotation in bbox_annotations:
            bbox_annotation = list(bbox_annotation)
            prim_path = bbox_annotation[1]
            name = bbox_annotation[1].split("/")[-1]
            corners = bbox_annotation[13]
            bboxes[name] = corners

        # Pick an object
        for obj in objs_of_interest:
            # Get the corners of the object's bbox
            bbox_vertices = bboxes[obj.name]
            bbox_vertices_heterogeneous = np.concatenate([bbox_vertices.T, np.ones((1, len(bbox_vertices)))], axis=0)

            # Get the image coordinates of each vertex
            projected_points_heterogeneous = view_proj_matrix @ bbox_vertices_heterogeneous
            projected_points = projected_points_heterogeneous[:2] / projected_points_heterogeneous[2:3]

            points_valid = (
                np.all(projected_points >= 0, axis=0)
                & (projected_points[0] < resolution["width"])
                & (projected_points[1] < resolution["height"])
            )
            if np.count_nonzero(points_valid) >= min_bbox_vertices_in_fov:
                return True

        return False

    return filter_fn


def no_relevant_object_in_img_filter(target_state, threshold=0.2):
    def filter_fn(env, cam, objs_of_interest):
        seg = env.simulator.renderer.render(modes="ins_seg")[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(seg)

        obj_body_ids = [x for obj in objs_of_interest for x in obj.get_body_ids()]
        relevant = np.count_nonzero(np.isin(body_ids, obj_body_ids))
        return relevant / len(seg.flatten()) > threshold

        # Count how many pixels per object.
        # ctr = collections.Counter(body_ids.flatten())
        # if -1 in ctr:
        #     del ctr[-1]
        #
        # target_state_value = random.uniform(0, 1) < 0.5
        # target_pixel_count = 0
        # for body_id, pixel_count in ctr.items():
        #     obj = env.simulator.scene.objects_by_id[body_id]
        #     if target_state in obj.states:
        #         if obj.states[target_state].get_value() == target_state_value:
        #             target_pixel_count += pixel_count
        # return target_pixel_count / len(seg.flatten()) > threshold

    return filter_fn


def point_in_object_filter():
    def filter_fn(env, cam, objs_of_interest):
        # Camera position
        camera_params = cam.get_obs()["camera"]
        cam_pos = camera_params["pose"][:-1, -1]
        bbox_annotations = cam.get_obs()["bbox_3d"]
        for bbox_annotation in bbox_annotations:
            bbox_annotation = list(bbox_annotation)
            corners = bbox_annotation[13]
            if point_in_bbox(cam_pos, corners):
                return False
        return True

    return filter_fn

def point_in_bbox(point, bbox):
    """Check if a point is inside a bounding box.

    Args:
        point: 3D point.
        bbox: 3D bounding box.

    Returns:
        True if the point is inside the bounding box, False otherwise.
    """
    # Check if the point is inside the bounding box
    return np.all(point >= bbox.min(axis=0)) and np.all(point <= bbox.max(axis=0))
