# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from functools import cached_property

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh, sample_mesh_keypoints

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.COLLISION_SIMPLIFICATION = True
m.DEFAULT_SIMULATION_RESOLUTION = 10
m.SELF_COLLISION = False
m.DEFAULT_YOUNG_MODULUS = 100.0
m.DEFAULT_POISSON_RATIO = 0.49
m.DEFAULT_DAMPING_SCALE = 0.5
m.DEFAULT_FRICTION = 0.5


class DeformablePrim(GeomPrim):
    """
    Provides high level functions to deal with a deformable prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @relative_prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
            specified:

            scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
            mass (None or float): If specified, mass of this body in kg
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # Make sure flatcache is not being used -- if so, raise an error, since we lose most of our needed functionality
        # (such as R/W to specific particle states) when flatcache is enabled
        assert not gm.ENABLE_FLATCACHE, "Cannot use flatcache with DeformablePrim!"

        self._mass_api = (
            lazy.pxr.UsdPhysics.MassAPI(self._prim)
            if self._prim.HasAPI(lazy.pxr.UsdPhysics.MassAPI)
            else lazy.pxr.UsdPhysics.MassAPI.Apply(self._prim)
        )

        # Possibly set the mass / density
        if "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]

        success_flag = (
            lazy.omni.physx.scripts.deformableUtils.add_physx_deformable_body(
                stage=og.sim.stage,
                prim_path=self._prim.GetPath(),
                collision_simplification=m.COLLISION_SIMPLIFICATION,
                simulation_hexahedral_resolution=m.DEFAULT_SIMULATION_RESOLUTION,
                self_collision=m.SELF_COLLISION,
            )
        )
        assert success_flag

        # Create a deformable body material and set it on the deformable body
        deformable_material_path = f"{self._prim.GetPath()}/deformableBodyMaterial"
        lazy.omni.physx.scripts.deformableUtils.add_deformable_body_material(
            stage=og.sim.stage,
            path=deformable_material_path,
            youngs_modulus=m.DEFAULT_YOUNG_MODULUS,
            poissons_ratio=m.DEFAULT_POISSON_RATIO,
            damping_scale=m.DEFAULT_DAMPING_SCALE,
            dynamic_friction=m.DEFAULT_FRICTION,
        )

        lazy.omni.physx.scripts.physicsUtils.add_physics_material_to_prim(
            stage=og.sim.stage,
            prim=self._prim,
            materialPath=deformable_material_path,
        )

    @cached_property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object. For DeformablePrim, always return False.
        """
        return False
    
    def update_handles(self):
        # no handles to update
        pass

    @property
    def volume(self):
        mesh = mesh_prim_to_trimesh_mesh(self.prim, include_normals=False, include_texcoord=False, world_frame=True)
        return mesh.volume if mesh.is_volume else mesh.convex_hull.volume

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for a link!")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        # We have to read the mass directly in the deformable prim
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        # We have to set the mass directly in the deformable prim
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for DeformablePrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for DeformablePrim")

    @property
    def body_name(self):
        """
        Returns:
            str: Name of this body
        """
        return self.prim_path.split("/")[-1]
    
    def wake(self):
        # TODO (eric): Just a pass through for now.
        return