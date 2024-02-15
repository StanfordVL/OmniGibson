import traceback
import json
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.systems.system_base import get_system
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
import trimesh
import tqdm
from omnigibson.systems import import_og_systems

gm.HEADLESS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

def generate_box(box_half_extent):
    # The floor plane already exists
    # We just need to generate the side planes
    plane_centers = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
    ]) * box_half_extent
    for i, pc in enumerate(plane_centers):
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{i}",
            name=f"plane_{i}",
            z_position=0,
            size=box_half_extent[2],
            color=None,
            visible=False,

            # TODO: update with new PhysicsMaterial API
            # static_friction=static_friction,
            # dynamic_friction=dynamic_friction,
            # restitution=restitution,
        )

        plane_as_prim = XFormPrim(
            prim_path=plane.prim_path,
            name=plane.name,
        )
        
        # Build the plane orientation from the plane normal
        horiz_dir = pc - np.array([0, 0, box_half_extent[2]])
        plane_z = -1 * horiz_dir / np.linalg.norm(horiz_dir)
        plane_x = np.array([0, 0, 1])
        plane_y = np.cross(plane_z, plane_x)
        plane_mat = np.array([plane_x, plane_y, plane_z]).T
        plane_quat = R.from_matrix(plane_mat).as_quat()
        plane_as_prim.set_position_orientation(pc, plane_quat)

def generate_particles_in_box(box_half_extent):
    water = get_system("water")
    particle_radius = water.particle_radius

    # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
    # and generate a grid of points based on the sampling distance
    low = np.array([-1, -1, 0]) * box_half_extent
    high = np.array([1, 1, 2]) * box_half_extent
    extent = np.ones(3) * box_half_extent * 2
    # We sample the range of each extent minus
    sampling_distance = 2 * particle_radius
    n_particles_per_axis = (extent / sampling_distance).astype(int)
    assert np.all(n_particles_per_axis), f"box is too small to sample any particle of radius {particle_radius}."

    # 1e-10 is added because the extent might be an exact multiple of particle radius
    arrs = [np.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2)
            for l, h, n in zip(low, high, n_particles_per_axis)]
    # Generate 3D-rectangular grid of points
    particle_positions = np.stack([arr.flatten() for arr in np.meshgrid(*arrs)]).T

    water.generate_particles(
        positions=particle_positions,
    )

    return water

def draw_mesh(mesh, parent_pos):
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    edge_vert_idxes = mesh.edges_unique
    N = len(edge_vert_idxes)
    colors = [(1., 0., 0., 1.) for _ in range(N)]
    sizes = [1. for _ in range(N)]
    points1 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 0]] + parent_pos).tolist()]
    points2 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 1]] + parent_pos).tolist()]
    draw.draw_lines(points1, points2, colors, sizes)

def process_object(cat, mdl, out_path):
    if og.sim:
        og.sim.clear()

    cfg = {
        "scene": {
            "type": "Scene",
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

    # Fix all joints to upper position
    for joint in fillable.joints.values():
        if joint.has_limit:
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()

    # Now move it into position
    aabb_extent = fillable.aabb_extent
    obj_bbox_height = aabb_extent[2]
    obj_bbox_center = fillable.aabb_center
    obj_bbox_bottom = obj_bbox_center - np.array([0, 0, obj_bbox_height / 2])
    obj_current_pos = fillable.get_position()
    obj_pos_wrt_bbox = obj_current_pos - obj_bbox_bottom
    obj_dipped_pos = obj_pos_wrt_bbox
    obj_free_pos = obj_pos_wrt_bbox + np.array([0, 0, 2 * aabb_extent[2] + 0.2])
    fillable.set_position_orientation(obj_free_pos, np.array([0, 0, 0, 1]))
    og.sim.step()

    # Now generate the box and the particles
    box_half_extent = aabb_extent * 1.5  # 1.5 times the space
    box_half_extent[2] = np.maximum(box_half_extent[2], 0.1)  # at least 10cm water
    generate_box(box_half_extent)
    og.sim.step()
    water = generate_particles_in_box(box_half_extent)
    for _ in range(100):
        og.sim.step()

    # Move the object down into the box slowly
    lin_vel = 0.05
    while True:
        delta_z = -lin_vel * og.sim.get_rendering_dt()
        cur_pos = fillable.get_position()
        new_pos = cur_pos + np.array([0, 0, delta_z])
        fillable.set_position(new_pos)
        og.sim.step()
        if fillable.get_position()[2] < obj_dipped_pos[2]:
            break

    # Let the particles settle
    for _ in range(100):
        og.sim.step()

    # Now move the object out of the water
    while True:
        delta_z = lin_vel * og.sim.get_rendering_dt()
        cur_pos = fillable.get_position()
        new_pos = cur_pos + np.array([0, 0, delta_z])
        fillable.set_position(new_pos)
        og.sim.step()
        if fillable.get_position()[2] > obj_free_pos[2]:
            break

    # Gentle side-by-side shakeoff
    spill_fraction = 0.1
    extents = aabb_extent[:2]
    # formula for how much to rotate for total spill to be spill_fraction of the volume
    angles = np.arctan(0.5 * spill_fraction * extents / aabb_extent[2])
    angles = np.flip(angles)

    print("Rotation amounts (degrees): ", np.rad2deg(angles))

    rotations = np.array([np.eye(3)[i] * angle * side for i, angle in enumerate(angles) for side in [-1, 1]])
    for r in rotations:
        total_steps = 60
        for _ in range(total_steps):
            delta_orn = R.from_euler("xyz", r / total_steps)
            cur_rot = R.from_quat(fillable.get_orientation())
            new_rot = delta_orn * cur_rot
            fillable.set_orientation(new_rot.as_quat())
            og.sim.step()
        for _ in range(30):
            og.sim.step()
        for _ in range(total_steps):
            delta_orn = R.from_euler("xyz", -r / total_steps)
            cur_rot = R.from_quat(fillable.get_orientation())
            new_rot = delta_orn * cur_rot
            fillable.set_orientation(new_rot.as_quat())
            og.sim.step()

    # Let the particles settle
    for _ in range(30):
        og.sim.step()

    # Get the particles whose center is within the object's AABB
    aabb_min, aabb_max = fillable.aabb
    particles = water.get_particles_position_orientation()[0]
    particle_point_offsets = np.array([e * side * water.particle_radius for e in np.eye(3) for side in [-1, 1]] + [np.zeros(3)])
    points = np.repeat(particles, len(particle_point_offsets), axis=0) + np.tile(particle_point_offsets, (len(particles), 1))
    points = points[np.all(points <= aabb_max, axis=1)]
    points = points[np.all(points >= aabb_min, axis=1)]
    assert len(points) > 0, "No points found in the AABB of the object."

    # Get the particles in the frame of the object
    points -= fillable.get_position()

    # Get the convex hull of the particles
    hull = trimesh.convex.convex_hull(points)

    # Save it somewhere
    hull.export(out_path, file_type="obj", include_normals=False, include_color=False, include_texture=False)

    # draw_mesh(hull, fillable.get_position())

    # from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand
    # container_prim_path = fillable.root_link.prim_path + "/container"
    # CreateMeshPrimWithDefaultXformCommand(prim_path=container_prim_path, prim_type="Mesh", trimesh_mesh=hull).do()
    # mesh_prim = XFormPrim(name="container", prim_path=container_prim_path)


def main():
    import sys, pathlib

    dataset_root = str(pathlib.Path(sys.argv[1]))
    gm.DATASET_PATH = str(dataset_root)

    # This is a hacky fix for systems not being loaded because of the dataset
    # path being changed later.
    import_og_systems()

    batch = sys.argv[2:]
    for path in batch:
        obj_category, obj_model = pathlib.Path(path).parts[-2:]
        obj_dir = pathlib.Path(dataset_root) / "objects" / obj_category / obj_model
        assert obj_dir.exists()
        print(f"Processing {path}")
        out_path = obj_dir / "fillable.obj"
        try:
            process_object(obj_category, obj_model, out_path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
