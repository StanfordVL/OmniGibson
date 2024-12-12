import sys

import numpy as np

sys.path.append(r"D:\ig_pipeline")

import json
import pathlib
from collections import Counter, defaultdict

import pymxs

import b1k_pipeline.utils
import trimesh.transformations

rt = pymxs.runtime

OUTPUT_FILENAME = "object_list.json"
COMPLEX_FINGERPRINT = False


def compute_moment_of_inertia(triangles, reference_point=None):
    """
    Compute the moment of inertia matrix for a triangle mesh around a reference point.

    Parameters:
    triangles: numpy array of shape (n, 3, 3) where n is the number of triangles,
              and each triangle is defined by 3 points with x,y,z coordinates
    reference_point: point around which to compute the moment of inertia (default: mesh centroid)

    Returns:
    numpy array of shape (3, 3) representing the moment of inertia matrix
    """

    assert (
        triangles.ndim == 3 and triangles.shape[1] == 3 and triangles.shape[2] == 3
    ), f"Bad triangle shape {triangles.shape}"

    def triangle_area(triangle):
        """Compute area of a single triangle using cross product."""
        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2))

    def triangle_centroid(triangle):
        """Compute centroid of a triangle."""
        return np.mean(triangle, axis=0)

    # Compute areas and total surface area
    areas = np.array([triangle_area(tri) for tri in triangles])
    total_area = np.sum(areas)

    # Compute surface mass density
    surface_density = 1.0 / total_area

    # Compute centroids
    centroids = np.array([triangle_centroid(tri) for tri in triangles])
    assert centroids.shape == (
        triangles.shape[0],
        3,
    ), f"Bad centroids shape {centroids.shape}"

    # If no reference point provided, use mesh centroid
    if reference_point is None:
        reference_point = np.average(centroids, weights=areas, axis=0)
        assert reference_point.shape == (
            3,
        ), f"Bad reference point shape {reference_point.shape}"

    # Initialize moment of inertia matrix
    I = np.zeros((3, 3))

    # For each triangle
    for triangle, area, centroid in zip(triangles, areas, centroids):
        # Translate triangle to reference point
        translated_points = triangle - reference_point

        # Compute local moment of inertia around triangle centroid
        local_I = np.zeros((3, 3))

        # For each point in the triangle
        for point in translated_points:
            x, y, z = point
            # Contribution to moment of inertia matrix
            local_I[0, 0] += y * y + z * z
            local_I[1, 1] += x * x + z * z
            local_I[2, 2] += x * x + y * y
            local_I[0, 1] -= x * y
            local_I[0, 2] -= x * z
            local_I[1, 2] -= y * z

        # Make matrix symmetric
        local_I[1, 0] = local_I[0, 1]
        local_I[2, 0] = local_I[0, 2]
        local_I[2, 1] = local_I[1, 2]

        # Scale by area and surface density
        local_I *= area * surface_density / 12

        # Apply parallel axis theorem
        r = centroid - reference_point
        parallel_axis_term = (
            area * surface_density * (np.eye(3) * np.dot(r, r) - np.outer(r, r))
        )

        # Add to total moment of inertia
        I += local_I + parallel_axis_term

    return I, reference_point


def main():
    object_names = [x.name for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    matches = [b1k_pipeline.utils.parse_name(name) for name in object_names]
    nomatch = [name for name, match in zip(object_names, matches) if match is None]

    success = len(nomatch) == 0
    needed = sorted(
        {
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("meta_type")
        }
    )
    provided = sorted(
        {
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("bad") and not x.group("meta_type")
        }
    )
    counts = Counter(
        [
            x.group("category") + "-" + x.group("model_id")
            for x in matches
            if x is not None and not x.group("meta_type")
        ]
    )
    max_tree = [
        (x.name, str(rt.classOf(x)), x.parent.name if x.parent else None)
        for x in rt.objects
    ]

    meshes = sorted(
        name
        for match, name in zip(matches, object_names)
        if match is not None and not match.group("meta_type")
    )

    meta_links = defaultdict(set)

    # Store vertex/face counts for bad model matching
    pivots_and_mesh_parts = defaultdict(list)
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        # It needs to be a name-matching object
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue

        # Check if this is the 0th instance and the lower mesh and not a meta link and not bad
        if (
            match.group("instance_id") != "0"
            or match.group("joint_side") == "upper"
            or match.group("meta_type")
            or match.group("bad")
        ):
            continue

        # Otherwise store the vertex and face info for the model ID
        model_id = match.group("model_id")

        # Record the orientation of the base link
        pivot = (
            None
            if match.group("link_name") and match.group("link_name") != "base_link"
            else np.hstack(
                [b1k_pipeline.utils.mat2arr(obj.transform), [[0], [0], [0], [1]]]
            ).T
        )

        # Get vertices and faces into numpy arrays for conversion
        num_verts = rt.polyop.GetNumVerts(obj)
        num_faces = rt.polyop.GetNumFaces(obj)
        verts = None
        faces = None
        if COMPLEX_FINGERPRINT:
            verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(num_verts)])
            faces = (
                np.array(
                    rt.polyop.getFacesVerts(obj, rt.execute("#{1..%d}" % num_faces))
                )
                - 1
            )
            assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"
        pivots_and_mesh_parts[model_id].append(
            (pivot, verts, faces, num_verts, num_faces)
        )

    # Accumulate the vertex and face counts and the moments of inertia
    mesh_fingerprints = {}
    for model_id, parts in pivots_and_mesh_parts.items():
        # Flatten the faces into a single array of triangles (keeping track of original vertex counts)
        vertex_count = 0
        face_count = 0
        triangle_subs = []
        pivots = []
        for orientation, verts, faces, num_verts, num_faces in parts:
            vertex_count += num_verts
            face_count += num_faces
            if COMPLEX_FINGERPRINT:
                triangle_subs.append(verts[faces])
            if orientation is not None:
                pivots.append(orientation)

        # Grab the orientation
        assert (
            len(pivots) == 1
        ), f"Expected 1 orientation for {model_id}, got {len(pivots)}"
        pivot = pivots[0]

        flattened_moment_of_inertia = None
        pivot_to_centroid = None
        if COMPLEX_FINGERPRINT:
            triangles = np.concatenate(triangle_subs, axis=0)  # Flatten the triangles

            # Orient the mesh such that it is in the frame of the orientation
            world_to_pivot = np.linalg.inv(pivot)
            oriented_triangles = trimesh.transformations.transform_points(
                triangles.reshape(-1, 3), world_to_pivot
            ).reshape(triangles.shape)

            # Rescale the mesh around its bounding box to exactly fit the unit cube.
            min_bound = np.min(triangles.reshape(-1, 3), axis=0)
            max_bound = np.max(triangles.reshape(-1, 3), axis=0)
            center = (min_bound + max_bound) / 2
            assert center.shape == (3,), f"Bad center shape {center.shape}"
            scale = max_bound - min_bound
            assert scale.shape == (3,), f"Bad scale shape {scale.shape}"
            scale[np.isclose(scale, 0)] = 1.0  # avoid degenerate cases
            scaled_triangles = (oriented_triangles - center) / scale

            # Compute the moment of inertia
            I, centroid = compute_moment_of_inertia(scaled_triangles)
            assert centroid.shape == (3,), f"Bad centroid shape {centroid.shape}"
            assert I.shape == (3, 3), f"Bad moment of inertia shape {I.shape}"

            # Where would the pivot be in the scaled frame?
            pivot_scaled = -center / scale
            assert pivot_scaled.shape == (
                3,
            ), f"Bad scaled pivot shape {pivot_scaled.shape}"

            # Compute the pivot-to-centroid vector in the scaled frame
            pivot_to_centroid = (centroid - pivot_scaled).tolist()
            flattened_moment_of_inertia = I.flatten().tolist()

        # Store the vertex count, face count, moment of inertia and pivot-to-centroid vector as the mesh fingerprint
        mesh_fingerprints[model_id] = (
            vertex_count,
            face_count,
            flattened_moment_of_inertia,
            pivot_to_centroid,
        )

    # Find meta links
    for obj, _, parent in max_tree:
        if not parent:
            continue
        parent_match = b1k_pipeline.utils.parse_name(parent)
        obj_match = b1k_pipeline.utils.parse_name(obj)
        if not parent_match or not obj_match:
            continue
        if not obj_match.group("meta_type") or parent_match.group("bad"):
            continue
        parent_id = (
            parent_match.group("category") + "-" + parent_match.group("model_id")
        )
        meta_type = obj_match.group("meta_type")
        meta_links[parent_id].add(meta_type)

    # Find joints
    for obj, _, _ in max_tree:
        obj_match = b1k_pipeline.utils.parse_name(obj)
        if not obj_match:
            continue
        if obj_match.group("joint_type") not in ["R", "P"]:
            continue
        parent_id = obj_match.group("category") + "-" + obj_match.group("model_id")
        meta_links[parent_id].add("joint")

    meta_links = {k: sorted(v) for k, v in sorted(meta_links.items())}

    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    results = {
        "success": success,
        "needed_objects": needed,
        "provided_objects": provided,
        "meshes": meshes,
        "meta_links": meta_links,
        "max_tree": max_tree,
        "object_counts": counts,
        "mesh_fingerprints": mesh_fingerprints,
        "error_invalid_name": sorted(nomatch),
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print("success:", success)
    print("error_invalid_name:", sorted(nomatch))


if __name__ == "__main__":
    main()
