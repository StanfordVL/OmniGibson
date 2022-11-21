# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omnigibson.utils.usd_utils import array_to_vtarray
from pxr import UsdPhysics, Gf
from pxr.Sdf import ValueTypeNames as VT

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.usd import get_shader_from_material
from omni.physx.scripts import particleUtils

from omnigibson.prims.geom_prim import GeomPrim
import numpy as np



DEBUG = False


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
        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        self._mass_api = UsdPhysics.MassAPI(self._prim) if self._prim.HasAPI(UsdPhysics.MassAPI) else \
            UsdPhysics.MassAPI.Apply(self._prim)

        # Possibly set the mass / density
        if "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]

        # TODO (eric): customize stiffness
        stretch_stiffness = 10000.0
        bend_stiffness = 200.0
        shear_stiffness = 100.0
        damping = 0.2
        particleUtils.add_physx_particle_cloth(
            stage=get_current_stage(),
            path=self._prim.GetPath(),
            dynamic_mesh_path=None,
            particle_system_path=f"/World/ClothSystem",
            spring_stretch_stiffness=stretch_stiffness,
            spring_bend_stiffness=bend_stiffness,
            spring_shear_stiffness=shear_stiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )

    def _initialize(self):
        super()._initialize()
        # TODO (eric): hacky way to get cloth rendering to work (otherwise, there exist some rendering artifacts).
        self._prim.CreateAttribute("primvars:isVolume", VT.Bool, False).Set(True)
        self._prim.GetAttribute("primvars:isVolume").Set(False)

        self.area_unfolded, self.diagonal_unfolded = self.calculate_projection_area_and_diagonal_unfolded()

    @property
    def particle_positions(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        import omnigibson.utils.transform_utils as T

        R = T.quat2mat(self.get_orientation())
        t = self.get_position()
        s = self.scale

        p_local = self.get_attribute(attr="points")
        p_world = (R @ (p_local * s).T).T + t

        return p_world

    @particle_positions.setter
    def particle_positions(self, pos):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        assert pos.shape[0] == self.particle_positions.shape[0], \
            f"Got mismatch in particle setting size: {pos.shape[0]}, vs. number of particles {self.particle_positions.shape[0]}!"
        pos = (pos - self.get_position()).astype(float)
        self.set_attribute(attr="points", val=array_to_vtarray(arr=pos, element_type=Gf.Vec3f))

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        pass

    @property
    def volume(self):
        raise NotImplementedError("Cannot get volume for ClothPrim")

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume for ClothPrim")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for ClothPrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for ClothPrim")

    def set_linear_velocity(self, velocity):
        """Sets the linear velocity of the prim in stage.

        Args:
            velocity (np.ndarray): linear velocity to set the rigid prim to. Shape (3,).
        """
        # TODO (eric): Just a pass through for now.
        return

    def set_angular_velocity(self, velocity):
        """Sets the angular velocity of the prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        # TODO (eric): Just a pass through for now.
        return


    def wake(self):
        """
        Enable physics for this rigid body
        """
        # TODO (eric): Just a pass through for now.
        return


    def calculate_projection_area_and_diagonal(self, dims=[0, 1], plot=DEBUG):
        """Calculate the projection area and the diagonal length when projecting to axis
        """
        points = self.particle_positions[:, dims]

        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        hull = ConvexHull(points)

        diagonal = 0
        for i in range(len(hull.vertices)):
            dist = np.sqrt(np.sum((points[hull.vertices] - points[hull.vertices[i]])**2, axis=1))
            diagonal = max(diagonal, np.max(dist))

        if plot:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.set_aspect('equal')

            plt.plot(points[:, 0], points[:, 1], 'o')
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
            plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
            plt.show()

        return hull.area, diagonal

    def calculate_projection_area_and_diagonal_unfolded(self):
        """Calculate the projection area and the diagonal length of the cloth in an unfolded state
        """

        # use the largest projection area as the unfolded area
        area_unfolded = 0.
        dims = [[0, 1], [0, 2], [1, 2]]

        for i in range(len(dims)):
            area, diagonal = self.calculate_projection_area_and_diagonal(dims=dims[i])
            if area > area_unfolded:
                area_unfolded = area
                diagonal_unfolded = diagonal

        return area_unfolded, diagonal_unfolded

    def calculate_smoothness(self, normal_z_percentage, normal_z_angle_diff):
         """Calculate the smoothness of the cloth according to the normal vectors

         Args:
             normal_z_percentage (float): a least what percentage of normal vectors should point to the gravity direction
             normal_z_angle_diff (float): rad, the threshold within which the normal vector is considered to be in the gravity direction
         """
         eps = 1e-6
         faceVertexCounts = self.get_attribute("faceVertexCounts")
         faceVertexIndices = self.get_attribute("faceVertexIndices")

         points = self.particle_positions[faceVertexIndices].reshape((len(faceVertexIndices) // 3, 3, 3))

         v1 = points[:, 2] - points[:, 0]
         v2 = points[:, 1] - points[:, 0]
         normal = np.cross(v1, v2)
         normal = normal / (np.linalg.norm(normal, ord=2, axis=1)[:, None] + eps)

         # projection on the gravity direction
         proj = np.abs(np.dot(normal, np.array([0., 0., 1])))

         # calculate the percentage of normal vectors that are on the gravity direction
         percentage = np.sum(proj > np.cos(normal_z_angle_diff)) / len(proj)

         return percentage > normal_z_percentage

    def folded(self):
        """Function to determine whether the cloth is folded or not
        """
        # Criterial #1: the threshold on the area reduction ratio of the convex hull of the projection on the XY plane
        AREA_REDUCTION_THRESHOLD = 0.5

        # Criterial #2: the threshold on the reduction of the diagonal of the projection's convex hull
        DIAGONAL_REDUCTION_THRESHOLD = 0.6

        # Criterial #3: smoothness, i.e., the percentage of the normal vectors that lie in the z (gravity) direction
        NORMAL_Z_PERCENTAGE = 0.5
        NORMAL_Z_ANGLE_DIFF = np.deg2rad(30.) # normal direction threshold, within which, it still considers to be along the z direction


        ### calculate the area and the diagonal of the current state
        area, diagonal = self.calculate_projection_area_and_diagonal()

        # check area reduction ratio
        flag_area_reduction = (area / self.area_unfolded) < AREA_REDUCTION_THRESHOLD

        # check the diagonal reduction ratio
        flag_diagonal_reduction = (diagonal / self.diagonal_unfolded) < DIAGONAL_REDUCTION_THRESHOLD

        # check the smoothness of the cloth
        flag_smoothness = self.calculate_smoothness(NORMAL_Z_PERCENTAGE, NORMAL_Z_ANGLE_DIFF)


        ### only check area and diagonal reduction for now
        # folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        folded = flag_area_reduction and flag_diagonal_reduction
        return folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness

