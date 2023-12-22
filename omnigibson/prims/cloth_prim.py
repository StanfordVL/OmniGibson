# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omnigibson.utils.geometry_utils import get_particle_positions_from_frame, get_particle_positions_in_frame
from pxr import UsdPhysics, Gf, Vt, PhysxSchema
from pxr.Sdf import ValueTypeNames as VT

from omni.isaac.core.prims import ClothPrimView

from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.systems import get_system
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import array_to_vtarray, mesh_prim_to_trimesh_mesh, sample_mesh_keypoints
from omnigibson.utils.constants import GEOM_TYPES
from omnigibson.utils.python_utils import classproperty
import omnigibson as og

import numpy as np
import torch


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Subsample cloth particle points to boost performance
m.N_CLOTH_KEYPOINTS = 1000
m.KEYPOINT_COVERAGE_THRESHOLD = 0.80
m.N_CLOTH_KEYFACES = 500



class ClothPrim(GeomPrim):
    """
    Provides high level functions to deal with a cloth prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Notes: if the prim does not already have a cloth api applied to it before it is loaded,
        it will apply it.

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
            specified:

            scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
            mass (None or float): If specified, mass of this body in kg
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Internal vars stored
        self._keypoint_idx = None
        self._keyface_idx = None
        self._cloth_prim_view = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # TODO: Do this using the cloth API too!
        self._mass_api = UsdPhysics.MassAPI(self._prim) if self._prim.HasAPI(UsdPhysics.MassAPI) else \
            UsdPhysics.MassAPI.Apply(self._prim)

        # Possibly set the mass / density
        if "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]

        # Clothify this prim, which is assumed to be a mesh
        ClothPrim.cloth_system.clothify_mesh_prim(mesh_prim=self._prim)

        # Track generated particle count. This is the only time we use the USD API.
        self._n_particles = len(self._prim.GetAttribute("points").Get())

        # Load the cloth prim view
        self._cloth_prim_view = ClothPrimView(self._prim_path)

        # positions = self.get_particle_positions()

        # # Sample mesh keypoints / keyvalues and sanity check the AABB of these subsampled points vs. the actual points
        # success = False
        # for i in range(10):
        #     self._keypoint_idx, self._keyface_idx = sample_mesh_keypoints(
        #         mesh_prim=self._prim,
        #         n_keypoints=m.N_CLOTH_KEYPOINTS,
        #         n_keyfaces=m.N_CLOTH_KEYFACES,
        #         seed=i,
        #     )

        #     keypoint_positions = positions[self._keypoint_idx]
        #     keypoint_aabb = keypoint_positions.min(axis=0), keypoint_positions.max(axis=0)
        #     true_aabb = positions.min(axis=0), positions.max(axis=0)
        #     overlap_vol = max(min(true_aabb[1][0], keypoint_aabb[1][0]) - max(true_aabb[0][0], keypoint_aabb[0][0]), 0) * \
        #         max(min(true_aabb[1][1], keypoint_aabb[1][1]) - max(true_aabb[0][1], keypoint_aabb[0][1]), 0) * \
        #         max(min(true_aabb[1][2], keypoint_aabb[1][2]) - max(true_aabb[0][2], keypoint_aabb[0][2]), 0)
        #     true_vol = np.product(true_aabb[1] - true_aabb[0])
        #     if overlap_vol / true_vol > m.KEYPOINT_COVERAGE_THRESHOLD:
        #         success = True
        #         break
        # assert success, f"Did not adequately subsample keypoints for cloth {self.name}!"

    def _initialize(self):
        super()._initialize()

        # Update the handles so that we can access particles
        self.update_handles()

        # TODO (eric): hacky way to get cloth rendering to work (otherwise, there exist some rendering artifacts).
        self._prim.CreateAttribute("primvars:isVolume", VT.Bool, False).Set(True)
        self._prim.GetAttribute("primvars:isVolume").Set(False)

        # Store the default position of the points in the local frame
        self._default_positions = get_particle_positions_in_frame(
            *self.get_position_orientation(), self.scale, self.get_particle_positions())

    @classproperty
    def cloth_system(cls):
        return get_system("cloth")

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles owned by this cloth prim
        """
        return self._n_particles

    @property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object. For ClothPrim, always return False.
        """
        return False

    def get_particle_positions(self, idxs=None):
        """
        Compute individual particle positions for this cloth prim

        Args:
            idxs (n-array or None): If set, will only calculate the requested indexed particle state

        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        all_particle_positions = self._cloth_prim_view.get_world_positions(clone=False)[0, :, :].cpu().numpy()
        return all_particle_positions[:self._n_particles] if idxs is None else all_particle_positions[idxs]

    def set_particle_positions(self, positions, idxs=None):
        """
        Sets individual particle positions for this cloth prim

        Args:
            positions (n-array): (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
            idxs (n-array or None): If set, will only set the requested indexed particle state
        """
        n_expected = self._n_particles if idxs is None else len(idxs)
        assert len(positions) == n_expected, \
            f"Got mismatch in particle setting size: {len(positions)}, vs. number of expected particles {n_expected}!"
        
        # First, get the particle positions.
        cur_pos = self._cloth_prim_view.get_world_positions().cpu().numpy()

        # Then apply the new positions at the appropriate indices
        cur_pos[0, idxs] = torch.Tensor(positions, device=cur_pos.device)

        # Then set to that position
        self._cloth_prim_view.set_world_positions(cur_pos)

    @property
    def keypoint_idx(self):
        """
        Returns:
            n-array: (N,) array specifying the keypoint particle IDs
        """
        return self._keypoint_idx

    @property
    def keyface_idx(self):
        """
        Returns:
            n-array: (N,) array specifying the keyface IDs
        """
        return self._keyface_idx

    @property
    def faces(self):
        """
        Grabs particle indexes defining each of the faces for this cloth prim

        Returns:
             np.array: (N, 3) numpy array, where each of the N faces are defined by the 3 particle indices
                corresponding to that face's vertices
        """
        return np.array(self.get_attribute("faceVertexIndices")).reshape(-1, 3)

    @property
    def keyfaces(self):
        """
        Grabs particle indexes defining each of the keyfaces for this cloth prim.
        Total number of keyfaces is m.N_CLOTH_KEYFACES

        Returns:
             np.array: (N, 3) numpy array, where each of the N keyfaces are defined by the 3 particle indices
                corresponding to that face's vertices
        """
        return self.faces[self._keyface_idx]

    @property
    def keypoint_particle_positions(self):
        """
        Grabs individual keypoint particle positions for this cloth prim.
        Total number of keypoints is m.N_CLOTH_KEYPOINTS

        Returns:
            np.array: (N, 3) numpy array, where each of the N keypoint particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        return self.get_particle_positions(idxs=self._keypoint_idx)

    @property
    def particle_velocities(self):
        """
        Grabs individual particle velocities for this cloth prim

        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        # the velocities attribute is w.r.t the world frame already
        return np.array(self.get_attribute(attr="velocities"))

    @particle_velocities.setter
    def particle_velocities(self, vel):
        """
        Set the particle velocities of this cloth

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame
        """
        assert vel.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {vel.shape[0]}, vs. number of particles {self._n_particles}!"

        # the velocities attribute is w.r.t the world frame already
        self.set_attribute(attr="velocities", val=Vt.Vec3fArray.FromNumpy(vel))

    def compute_face_normals(self, face_ids=None):
        """
        Grabs individual face normals for this cloth prim

        Args:
            face_ids (None or n-array): If specified, list of face IDs whose corresponding normals should be computed
                If None, all faces will be used

        Returns:
            np.array: (N, 3) numpy array, where each of the N faces' normals are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        faces = self.faces if face_ids is None else self.faces[face_ids]
        points = self.get_particle_positions(idxs=faces.flatten()).reshape(-1, 3, 3)
        return self.compute_face_normals_from_particle_positions(positions=points)

    def compute_face_normals_from_particle_positions(self, positions):
        """
        Grabs individual face normals for this cloth prim

        Args:
            positions (n-array): (N, 3, 3) array specifying the per-face particle positions

        Returns:
            np.array: (N, 3) numpy array, where each of the N faces' normals are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        # Shape [F, 3]
        v1 = positions[:, 2, :] - positions[:, 0, :]
        v2 = positions[:, 1, :] - positions[:, 0, :]
        normals = np.cross(v1, v2)
        return normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    def contact_list(self, keypoints_only=True):
        """
        Get list of all current contacts with this cloth body

        Args:
            keypoints_only (bool): If True, will only check contact with this cloth's keypoints

        Returns:
            list of CsRawData: raw contact info for this cloth body
        """
        contacts = []
        def report_hit(hit):
            contacts.append(CsRawData(
                time=0.0,  # dummy value
                dt=0.0,  # dummy value
                body0=self.prim_path,
                body1=hit.rigid_body,
                position=pos,
                normal=np.zeros(3),  # dummy value
                impulse=np.zeros(3),  # dummy value
            ))
            return True

        positions = self.keypoint_particle_positions if keypoints_only else self.get_particle_positions()
        for pos in positions:
            og.sim.psqi.overlap_sphere(ClothPrim.cloth_system.particle_contact_offset, pos, report_hit, False)

        return contacts

    def update_handles(self):
        assert og.sim._physics_sim_view._backend is not None, "Physics sim backend not initialized!"
        self._cloth_prim_view.initialize(og.sim.physics_sim_view)
        assert self._n_particles <= self._cloth_prim_view.max_particles_per_cloth, \
            f"Got more particles than the maximum allowed for this cloth! Got {self._n_particles}, max is " \
            f"{self._cloth_prim_view.max_particles_per_cloth}!"

    @property
    def volume(self):
        mesh = self.prim
        mesh_type = mesh.GetPrimTypeInfo().GetTypeName()
        assert mesh_type in GEOM_TYPES, f"Invalid collision mesh type: {mesh_type}"
        if mesh_type == "Mesh":
            # We construct a trimesh object from this mesh in order to infer its volume
            trimesh_mesh = mesh_prim_to_trimesh_mesh(mesh)
            mesh_volume = trimesh_mesh.volume if trimesh_mesh.is_volume else trimesh_mesh.convex_hull.volume
        elif mesh_type == "Sphere":
            mesh_volume = 4 / 3 * np.pi * (mesh.GetAttribute("radius").Get() ** 3)
        elif mesh_type == "Cube":
            mesh_volume = mesh.GetAttribute("size").Get() ** 3
        elif mesh_type == "Cone":
            mesh_volume = np.pi * (mesh.GetAttribute("radius").Get() ** 2) * mesh.GetAttribute("height").Get() / 3
        elif mesh_type == "Cylinder":
            mesh_volume = np.pi * (mesh.GetAttribute("radius").Get() ** 2) * mesh.GetAttribute("height").Get()
        else:
            raise ValueError(f"Cannot compute volume for mesh of type: {mesh_type}")

        mesh_volume *= np.product(self.get_world_scale())
        return mesh_volume

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for a link!")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        # We have to read the mass directly in the cloth prim
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        # We have to set the mass directly in the cloth prim
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for ClothPrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for ClothPrim")

    @property
    def body_name(self):
        """
        Returns:
            str: Name of this body
        """
        return self.prim_path.split("/")[-1]

    def get_linear_velocity(self):
        """
        Returns:
            np.ndarray: current average linear velocity of the particles of the cloth prim. Shape (3,).
        """
        return np.array(self._prim.GetAttribute("velocities").Get()).mean(axis=0)

    def get_angular_velocity(self):
        """
        Returns:
            np.ndarray: zero vector as a placeholder because a cloth prim doesn't have an angular velocity. Shape (3,).
        """
        return np.zeros(3)

    def set_linear_velocity(self, velocity):

        """
        Sets the linear velocity of all the particles of the cloth prim.

        Args:
            velocity (np.ndarray): linear velocity to set all the particles of the cloth prim to. Shape (3,).
        """
        vel = self.particle_velocities
        vel[:] = velocity
        self.particle_velocities = vel

    def set_angular_velocity(self, velocity):
        """
        Simply returns because a cloth prim doesn't have an angular velocity

        Args:
            velocity (np.ndarray): linear velocity to set all the particles of the cloth prim to. Shape (3,).
        """
        return

    def wake(self):
        # TODO (eric): Just a pass through for now.
        return

    @property
    def bend_stiffness(self):
        """
        Returns:
            float: spring bend stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springBendStiffness")

    @bend_stiffness.setter
    def bend_stiffness(self, bend_stiffness):
        """
        Args:
            bend_stiffness (float): spring bend stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springBendStiffness", bend_stiffness)

    @property
    def damping(self):
        """
        Returns:
            float: spring damping of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springDamping")

    @damping.setter
    def damping(self, damping):
        """
        Args:
            damping (float): spring damping of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springDamping", damping)

    @property
    def shear_stiffness(self):
        """
        Returns:
            float: spring shear_stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springShearStiffness")

    @shear_stiffness.setter
    def shear_stiffness(self, shear_stiffness):
        """
        Args:
            shear_stiffness (float): spring shear_stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springShearStiffness", shear_stiffness)

    @property
    def stretch_stiffness(self):
        """
        Returns:
            float: spring stretch_stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springStretchStiffness")

    @stretch_stiffness.setter
    def stretch_stiffness(self, stretch_stiffness):
        """
        Args:
            stretch_stiffness (float): spring stretch_stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springStretchStiffness", stretch_stiffness)

    @property
    def particle_group(self):
        """
        Returns:
            int: Particle group this instancer belongs to
        """
        return self.get_attribute(attr="physxParticle:particleGroup")

    @particle_group.setter
    def particle_group(self, group):
        """
        Args:
            group (int): Particle group this instancer belongs to
        """
        self.set_attribute(attr="physxParticle:particleGroup", val=group)

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()
        state["particle_group"] = self.particle_group
        state["n_particles"] = self.n_particles
        state["particle_positions"] = self.get_particle_positions()
        state["particle_velocities"] = self.particle_velocities
        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)
        # Sanity check the identification number and particle group
        assert self.particle_group == state["particle_group"], f"Got mismatch in particle group for this cloth " \
            f"when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."

        # Set values appropriately
        self._n_particles = state["n_particles"]
        for attr in ("positions", "velocities"):
            attr_name = f"particle_{attr}"
            # Make sure the loaded state is a numpy array, it could have been accidentally casted into a list during
            # JSON-serialization
            attr_val = np.array(state[attr_name]) if not isinstance(attr_name, np.ndarray) else state[attr_name]
            setattr(self, attr_name, attr_val)

    def _serialize(self, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        return np.concatenate([
            state_flat,
            [state["particle_group"], state["n_particles"]],
            state["particle_positions"].reshape(-1),
            state["particle_velocities"].reshape(-1),
        ]).astype(float)

    def _deserialize(self, state):
        # Run super first
        state_dict, idx = super()._deserialize(state=state)

        particle_group = int(state[idx])
        n_particles = int(state[idx + 1])

        # Sanity check the identification number
        assert self.particle_group == particle_group, f"Got mismatch in particle group for this particle " \
            f"instancer when deserializing state! Should be: {self.particle_group}, got: {particle_group}."

        # De-compress from 1D array
        state_dict["particle_group"] = particle_group
        state_dict["n_particles"] = n_particles

        # Process remaining keys and reshape automatically
        keys = ("particle_positions", "particle_velocities")
        sizes = ((n_particles, 3), (n_particles, 3))

        idx += 2
        for key, size in zip(keys, sizes):
            length = np.product(size)
            state_dict[key] = state[idx: idx + length].reshape(size)
            idx += length

        return state_dict, idx

    def reset(self):
        """
        Reset the points to their default positions in the local frame, and also zeroes out velocities
        """
        if self.initialized:
            self.set_particle_positions(get_particle_positions_from_frame(
                *self.get_position_orientation(), self.scale, self._default_positions))
            self.particle_velocities = np.zeros((self._n_particles, 3))
