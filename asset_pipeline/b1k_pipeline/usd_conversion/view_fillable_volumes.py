import hashlib
import os
import sys
import glob
import pathlib
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.systems.system_base import get_system
from omnigibson.utils.ui_utils import KeyboardEventHandler
import omnigibson.lazy as lazy
import trimesh
import json
import tqdm

gm.HEADLESS = False
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True

ASSIGNMENT_FILE = os.path.join(gm.DATASET_PATH, "fillable_assignments.json")

MAX_BBOX = 0.3

def get_assignments():
    if not os.path.exists(ASSIGNMENT_FILE):
        return {}

    with open(ASSIGNMENT_FILE, "r") as f:
        return json.load(f)
    
def add_assignment(mdl, assignment):
    assignments = get_assignments()
    assignments[mdl] = assignment
    with open(ASSIGNMENT_FILE, "w") as f:
        json.dump(assignments, f)

def draw_mesh(mesh, parent_pos, color=(1., 0., 0., 1.)):
    draw = lazy.omni.isaac.debug_draw._debug_draw.acquire_debug_draw_interface()
    edge_vert_idxes = mesh.edges_unique
    N = len(edge_vert_idxes)
    colors = [color for _ in range(N)]
    sizes = [1. for _ in range(N)]
    points1 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 0]] + parent_pos).tolist()]
    points2 = [tuple(x) for x in (mesh.vertices[edge_vert_idxes[:, 1]] + parent_pos).tolist()]
    draw.draw_lines(points1, points2, colors, sizes)

def generate_particles_in_mesh(mesh, parent_pos):
    water = get_system("water")
    particle_radius = water.particle_radius

    # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
    # and generate a grid of points based on the sampling distance
    
    low, high = mesh.bounds
    extent = high - low
    # We sample the range of each extent minus
    sampling_distance = 2 * particle_radius
    n_particles_per_axis = (extent / sampling_distance).astype(int)
    assert np.all(n_particles_per_axis), f"box is too small to sample any particle of radius {particle_radius}."

    # 1e-10 is added because the extent might be an exact multiple of particle radius
    arrs = [np.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2)
            for l, h, n in zip(low, high, n_particles_per_axis)]
    # Generate 3D-rectangular grid of points
    particle_positions = np.stack([arr.flatten() for arr in np.meshgrid(*arrs)]).T + parent_pos[None, :]

    # Remove the particles that are outside
    particle_positions = particle_positions[mesh.contains(particle_positions)]

    # Remove the particles that are colliding with the object
    # particle_positions = particle_positions[np.where(water.check_in_contact(particle_positions) == 0)[0]]

    water.generate_particles(
        positions=particle_positions,
    )

    return water

def view_object(cat, mdl):
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

    water = get_system("water")
    fillable = env.scene.object_registry("name", "fillable")
    fillable.set_position([0, 0, fillable.aabb_extent[2]])
    og.sim.step()

    # Reset keyboard bindings
    KeyboardEventHandler.KEYBOARD_CALLBACKS = {}

    print("\n\nNow processing:", cat, mdl)

    # Create the water resetter
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R,
        callback_fn=lambda: water.remove_all_particles(),
    )
    print("Press R to remove all water")

    # Create the stopper function
    keep_rendering = True
    def stop_rendering():
        nonlocal keep_rendering
        keep_rendering = False
    def save_assignment_and_stop(assignment):
        print(f"Chose option {assignment} for {cat}/{mdl}")
        add_assignment(mdl, assignment)
        stop_rendering()
    
    # Skip without any assignment
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.J,
        callback_fn=stop_rendering,
    )
    print("Press J to skip")

    # Skip with assignment that says Benjamin should fix
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.K,
        callback_fn=lambda: save_assignment_and_stop("fix"),
    )
    print("Press K to indicate object needs fixing to make fillable.")

    # Skip with assignment that says we should hand-annotate
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.L,
        callback_fn=lambda: save_assignment_and_stop("manual"),
    )
    print("Press L to indicate neither option works despite object being fillable (hand-annotate).")

    dip_path = pathlib.Path(__file__).parents[2] / "artifacts/parallels/fillable_volumes/objects" / cat / mdl / "fillable_dip.obj"
    if dip_path.exists():
        # Find the scale the mesh was generated at
        scale = np.minimum(1, MAX_BBOX / np.max(fillable.native_bbox))

        dip_mesh = trimesh.load(dip_path, force="mesh")
        inv_scale = 1 / scale
        transform = np.diag([inv_scale, inv_scale, inv_scale, 1])
        dip_mesh.apply_transform(transform)

        # Draw the mesh
        draw_mesh(dip_mesh, fillable.get_position(), color=(1., 0., 0., 1.))

        # Add the dip option filler
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Z,
            callback_fn=lambda: generate_particles_in_mesh(dip_mesh, fillable.get_position()),
        )
        print("Press Z to fill dip (red) with water")

        # Add the dip option chooser
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.X,
            callback_fn=lambda: save_assignment_and_stop("dip"),
        )
        print("Press X to choose the dip (red) option.")

    ray_path = pathlib.Path(__file__).parents[2] / "artifacts/parallels/fillable_volumes/objects" / cat / mdl / "fillable_ray.obj"
    if ray_path.exists():
        ray_mesh = trimesh.load(ray_path, force="mesh")

        # Draw the mesh
        draw_mesh(ray_mesh, fillable.get_position(), color=(0., 0., 1., 1.))

        # Add the ray option filler
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.A,
            callback_fn=lambda: generate_particles_in_mesh(ray_mesh, fillable.get_position()),
        )
        print("Press A to fill ray (blue) with water")

        # Add the ray option chooser
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S,
            callback_fn=lambda: save_assignment_and_stop("ray"),
        )
        print("Press S to choose the ray (blue) option.")


    while keep_rendering:
        og.sim.step()


def main():
    idx = int(sys.argv[1])
    idxes = int(sys.argv[2])

    # Get all models that have a fillable file.
    fillable_ids = glob.glob(os.path.join(gm.DATASET_PATH, "objects/*/*/fillable_*.obj"))
    fillables = sorted({tuple(pathlib.Path(fillable_id).parts[-3:-1]) for fillable_id in fillable_ids})

    # Get the ones that don't have a fillable assignment
    assignments = get_assignments()
    fillables = [(cat, mdl) for cat, mdl in fillables if mdl not in assignments]

    # Get the ones whose model hash match our ID
    fillables = [
        (cat, mdl)
        for cat, mdl in fillables
        if int(hashlib.md5(mdl.encode()).hexdigest(), 16) % idxes == idx
    ]

    for cat, mdl in tqdm.tqdm(fillables):
        view_object(cat, mdl)


if __name__ == "__main__":
    main()
