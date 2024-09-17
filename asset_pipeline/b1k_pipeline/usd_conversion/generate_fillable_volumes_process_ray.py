import traceback
import numpy as np
import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.systems.system_base import get_system
from omnigibson.utils.asset_utils import decrypted
from omnigibson.utils.sampling_utils import raytest_batch, raytest
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import trimesh
import shapely

gm.HEADLESS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

MAX_BBOX = 0.3

def draw_mesh(mesh, parent_pos):
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    edge_vert_idxes = mesh.edges_unique
    N = len(edge_vert_idxes)
    colors = [(1., 0., 0., 1.) for _ in range(N)]
    sizes = [1. for _ in range(N)]
    points1 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 0]] + parent_pos).tolist()]
    points2 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 1]] + parent_pos).tolist()]
    draw.draw_lines(points1, points2, colors, sizes)


def sample_radial_rays(point, n=40, dist=4.0):
    angles = np.arange(n) * 2 * np.pi / n
    x = np.cos(angles)
    y = np.sin(angles)
    start_points = np.ones((n, 3)) * point.reshape(1, 3)
    end_points = np.array([x, y, np.zeros(n)]).T * dist + point.reshape(1, 3)
    return raytest_batch(start_points, end_points)


def sample_raytest_start_end_full_grid_topdown(
    obj,
    ray_spacing,
    aabb_offset=None,
    aabb_offset_fraction=macros.utils.sampling_utils.DEFAULT_AABB_OFFSET_FRACTION,
):
    bbox_center = obj.aabb_center
    bbox_orn = np.array([0, 0, 0, 1.0])
    bbox_bf_extent = obj.aabb_extent
    aabb_offset = aabb_offset_fraction * bbox_bf_extent if aabb_offset is None else aabb_offset
    # bbox_center, bbox_orn, bbox_bf_extent, _ = obj.get_base_aligned_bbox(xy_aligned=True, fallback_to_aabb=True)

    half_extent_with_offset = (bbox_bf_extent / 2) + aabb_offset
    x = np.linspace(-half_extent_with_offset[0], half_extent_with_offset[0], int(half_extent_with_offset[0] * 2 / ray_spacing) + 1)
    y = np.linspace(-half_extent_with_offset[1], half_extent_with_offset[1], int(half_extent_with_offset[1] * 2 / ray_spacing) + 1)
    n_rays = len(x) * len(y)

    start_points = np.stack([
        np.tile(x, len(y)),
        np.repeat(y, len(x)),
        np.ones(n_rays) * half_extent_with_offset[2],
    ]).T

    end_points = np.copy(start_points)
    end_points[:, 2] = -half_extent_with_offset[2]

    # Convert the points into the world frame
    to_wf_transform = T.pose2mat((bbox_center, bbox_orn))
    start_points = trimesh.transformations.transform_points(start_points, to_wf_transform)
    end_points = trimesh.transformations.transform_points(end_points, to_wf_transform)

    start_points = np.expand_dims(start_points, axis=1)
    end_points = np.expand_dims(end_points, axis=1)

    return start_points, end_points


def sample_fillable_point_from_top(obj, n_rays, prop_successful):
    aabb_low, aabb_high = obj.aabb
    # Sample uniformly from the top of the AABB, and take the lowest z and randomly sample again
    start_rays, end_rays = sample_raytest_start_end_full_grid_topdown(obj,
                                                                      np.mean((aabb_high[:2] - aabb_low[:2]) / 10.0))
    down_results = raytest_batch(start_rays.reshape(-1, 3), end_rays.reshape(-1, 3))
    down_hit_results = np.array([result["position"] for result in down_results if result["hit"]])

    z = down_hit_results[:, 2].min() + 0.01
    point = down_hit_results[np.argmin(down_hit_results[:, 2])]
    point[2] += 0.01
    results = sample_radial_rays(point, n=n_rays)

    if np.mean([result["hit"] for result in results]) >= prop_successful and np.all(
        [result["distance"] > 0 for result in results if result["hit"]]):

        xs = np.array([result["position"][0] for result in results if result["hit"]])
        ys = np.array([result["position"][1] for result in results if result["hit"]])
        center = np.array([xs.mean(), ys.mean(), z])

        return True, center

    return False, np.zeros(3)


def process_object(cat, mdl, out_path):
    if og.sim:
        og.sim.clear()
    else:
        og.launch()

    # First get the native bounding box of the object
    usd_path = DatasetObject.get_usd_path(category=cat, model=mdl)
    usd_path = usd_path.replace(".usd", ".encrypted.usd")
    with decrypted(usd_path) as fpath:
        stage = lazy.pxr.Usd.Stage.Open(fpath)
        prim = stage.GetDefaultPrim()
        bounding_box = np.array(prim.GetAttribute("ig:nativeBB").Get())

    cfg = {
        "scene": {
            "type": "Scene",
            "use_floor_plane": False,
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "fillable",
                "category": cat,
                "model": mdl,
                "kinematic_only": False,
                "fixed_base": True,
            },
        ]
    }

    env = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")
    fillable.set_position([0, 0, 0])
    og.sim.step()

    # 1. - 2b. Start at center of AABB
    point = fillable.aabb_center
    aabb_center = fillable.aabb_center
    aabb_low, aabb_high = fillable.aabb
    avg_diameter = np.mean(fillable.aabb_extent[:2])
    success = False
    results = None
    n_rays = 40
    n_rays_half = int(n_rays / 2)
    n_rays_quarter = int(n_rays / 4)
    assert n_rays_half == n_rays / 2.0
    assert n_rays_quarter == n_rays / 4.0
    prop_successful = 0.75

    for i in range(200):
        results = sample_radial_rays(point, n=n_rays)
        if np.mean([result["hit"] for result in results]) >= prop_successful and np.all([result["distance"] > 0 for result in results if result["hit"]]):
            success = True
            break
        point = np.random.uniform(low=aabb_low, high=aabb_high)
        point[2] = aabb_center[2]

    if not success:
        # Try sampling from the top
        success, point = sample_fillable_point_from_top(fillable, n_rays, prop_successful)

    if success:
        # 2c. Move point to center of circle
        xs = np.array([result["position"][0] for result in results if result["hit"]])
        ys = np.array([result["position"][1] for result in results if result["hit"]])
        center = np.array([xs.mean(), ys.mean(), point[2]])
        # delta_x = (results[0]["distance"] - results[n_rays_half]["distance"]) / 2.0
        # delta_y = (results[n_rays_quarter]["distance"] - results[n_rays_half + n_rays_quarter]["distance"]) / 2.0
        # center = point + np.array([delta_x, delta_y, 0.0])

        # 3. Shoot ray up and down to hit boundaries
        down_ray = {"hit": False}

        while (not down_ray["hit"] or down_ray["distance"] == 0.0) and center[2] <= (aabb_high[2] + 0.01):
            center = center + np.array([0, 0, 0.01])
            down_ray = raytest(
                start_point=center,
                end_point=center - np.array([0, 0, 4.0]),
            )
        if not down_ray["hit"] or down_ray["distance"] == 0.0:
            success = False

    if success:
        results = sample_radial_rays(center, n=n_rays)
        if not (np.mean([result["hit"] for result in results]) >= prop_successful and np.all(
            [result["distance"] > 0 for result in results if result["hit"]])):

            success, center = sample_fillable_point_from_top(fillable, n_rays, prop_successful)
            assert success

        down_ray = raytest(
            start_point=center,
            end_point=center - np.array([0, 0, 4.0]),
        )

        assert down_ray["hit"]
        z_low = down_ray["position"][2]
        up_ray = raytest(
            start_point=center,
            end_point=center + np.array([0, 0, 4.0]),
        )
        z_high = up_ray["position"][2] if up_ray["hit"] else aabb_high[2]

        # 4. Sample points and record positions
        mesh_points = np.array([]).reshape(0, 3)
        z_dist = np.clip((aabb_high[2] - aabb_low[2]) / 8.0, 0.001, 0.05)
        offset = avg_diameter / 20.0
        min_diameter = avg_diameter / 5.0
        z = z_low + offset / 2.0
        layer_points = []
        layer_hit_positions = None
        i = 0
        while z <= z_high:
            layer_center = np.array([center[0], center[1], z])
            layer_results = sample_radial_rays(point=layer_center, n=n_rays)
            layer_hit_positions = np.array([result["position"] for result in layer_results if result["hit"]])
            if not len(layer_hit_positions) / len(layer_results) >= prop_successful:
                if i <= 1:
                    # We failed to sample, probably a degenerate mesh, fail
                    success = False
                break
            # If the distance between points is less than a threshold, also break
            # assert layer_results[0]["distance"] != layer_results[n_rays_half]["distance"]
            # if (layer_results[0]["distance"] + layer_results[n_rays_half]["distance"]) < min_diameter:
            if len(mesh_points) > 0 and np.any((layer_hit_positions.max(axis=0)[:2] - layer_hit_positions.min(axis=0)[:2]) < min_diameter):
                break
            layer_rays = (layer_hit_positions - layer_center.reshape(1, 3))
            layer_points = layer_hit_positions - layer_rays * offset / np.linalg.norm(layer_rays)
            # layer_points = [np.array(result["position"]) - (result["position"] - layer_center) * offset / np.linalg.norm(
            #     (result["position"] - layer_center)) for result in layer_results]
            mesh_points = np.concatenate([mesh_points, layer_points], axis=0)
            z += z_dist
            i += 1

        # Prune previous layer
        mesh_points = mesh_points[:-len(layer_points)]

    if not success:
        print("Failed to find sufficient cavity to generate fillable, skipping.")
        og.sim.stop()
        return

    # 5. Create trimesh
    # print(f"n mesh points: {len(mesh_points)}")
    tm = trimesh.Trimesh(vertices=np.array(mesh_points))

    # 6. Create convex hull
    hull = tm.convex_hull
    hull.unmerge_vertices()

    # 7. Take the xy-plane projection of the convex hull, and then sample rays shooting downwards to compensate for the
    # original z offset
    # We know the bounding box is [1, 1, 1], so sample points uniformly in XY plane
    proj = trimesh.path.polygons.projected(hull, normal=[0, 0, 1])
    n_dim_samples = 10
    x_range = np.linspace(aabb_low[0], aabb_high[0], n_dim_samples)
    y_range = np.linspace(aabb_low[1], aabb_high[1], n_dim_samples)
    ray_grid = np.dstack(np.meshgrid(x_range, y_range, indexing="ij"))
    ray_grid_flattened = ray_grid.reshape(-1, 2)

    # Check which rays are within the polygon
    is_within = proj.contains(shapely.points(ray_grid_flattened))
    xy_ray_positions = ray_grid_flattened[is_within]

    # Shoot these rays downwards and record their poses -- add them to the point set
    z_range = np.linspace(aabb_low[2], aabb_high[2], n_dim_samples)
    additional_points = []
    for xy_ray_pos in xy_ray_positions:
        # Find a corresponding z value within the convex hull to use as the start raycasting point
        start_samples = np.zeros((n_dim_samples, 3))
        start_samples[:, :2] = xy_ray_pos
        start_samples[:, 2] = z_range
        is_contained = hull.contains(start_samples)
        if not np.any(is_contained):
            # Just skip this sample
            continue
        # Use the lowest point (i.e.: the first idx that is True) as the raycasting start point
        z = z_range[np.where(is_contained)[0][0]]

        # Raycast downwards and record the hit point
        start = np.array([*xy_ray_pos, z])
        end = np.array(start)
        end[2] = aabb_low[2]
        down_ray = raytest(
            start_point=start,
            end_point=end,
        )
        # If we have a valid hit with nonzero distance, record this point
        if down_ray["hit"] and down_ray["distance"] > 0:
            additional_points.append(down_ray["position"])

    # Append all additional points to our existing set of points
    mesh_points = np.concatenate([mesh_points, np.array(additional_points)], axis=0)

    # Denormalize the mesh points based on the objects' scale
    scale = 1.0 / fillable.scale
    mesh_points = mesh_points * scale.reshape(1, 3)

    # Re-write to trimesh and take the finalized convex hull
    tm = trimesh.Trimesh(vertices=np.array(mesh_points))

    # 6. Create convex hull
    # c_tms = trimesh.decomposition.convex_decomposition(tm)
    hull = tm.convex_hull
    hull.unmerge_vertices()

    # # Get the collision meshes, subtract them from the hull
    # collision_trimeshes = [
    #     mesh_prim_to_trimesh_mesh(cm.prim).apply_transform(np.array(lazy.omni.isaac.core.utils.xforms._get_world_pose_transform_w_scale(cm.prim_path)).T)
    #     for rb in fillable.links.values()
    #     for cm in rb.collision_meshes.values()
    # ]
    # for cm in collision_trimeshes:
    #     hull = trimesh.boolean.difference([hull, cm], engine="manifold")

    # # Get the convex hull again
    # hull = tm.convex_hull

    # Save it somewhere
    hull.export(out_path, file_type="obj", include_normals=False, include_color=False, include_texture=False)

    # draw_mesh(hull, fillable.get_position())
    # while True:
    #     og.sim.render()

    # from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand
    # container_prim_path = fillable.root_link.prim_path + "/container"
    # CreateMeshPrimWithDefaultXformCommand(prim_path=container_prim_path, prim_type="Mesh", trimesh_mesh=hull).do()
    # mesh_prim = XFormPrim(name="container", prim_path=container_prim_path)


def main():
    import sys, pathlib

    dataset_root = str(pathlib.Path(sys.argv[1]))
    gm.DATASET_PATH = str(dataset_root)

    batch = sys.argv[2:]
    for path in batch:
        obj_category, obj_model = pathlib.Path(path).parts[-2:]
        obj_dir = pathlib.Path(dataset_root) / "objects" / obj_category / obj_model
        assert obj_dir.exists()
        print(f"Processing {path}")
        out_path = obj_dir / "fillable_ray.obj"
        try:
            process_object(obj_category, obj_model, out_path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
