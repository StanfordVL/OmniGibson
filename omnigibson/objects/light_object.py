import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class LightObject(StatefulObject):
    """
    LightObjects are objects that generate light in the simulation
    """

    LIGHT_TYPES = {
        "Cylinder",
        "Disk",
        "Distant",
        "Dome",
        "Geometry",
        "Rect",
        "Sphere",
    }

    def __init__(self, config: LightObjectConfig):
        """
        Args:
            config (LightObjectConfig): Configuration object containing all parameters for this light object
        """
        # Make sure primitive type is valid
        assert_valid_key(key=config.light_type, valid_keys=self.LIGHT_TYPES, name="light_type")
        
        # Store the config
        self._config = config
        
        # Other attributes to be filled in at runtime
        self._light_link = None

        # Run super method
        super().__init__(config=config)

    def _load(self):
        # Define XForm and base link for this light
        prim = og.sim.stage.DefinePrim(self.prim_path, "Xform")
        og.sim.stage.DefinePrim(f"{self.prim_path}/base_link", "Xform")

        # Define the actual light link
        (
            getattr(lazy.pxr.UsdLux, f"{self.light_type}Light")
            .Define(og.sim.stage, f"{self.prim_path}/base_link/light")
            .GetPrim()
        )

        return prim

    def _post_load(self):
        # run super first
        super()._post_load()

        # Grab reference to light link
        self._light_link = XFormPrim(
            relative_prim_path=f"{self._relative_prim_path}/base_link/light", name=f"{self.name}:light_link"
        )
        self._light_link.load(self.scene)

        # Apply Shaping API and set default cone angle attribute
        lazy.pxr.UsdLux.ShapingAPI.Apply(self._light_link.prim).GetShapingConeAngleAttr().Set(180.0)

        # Set the intensity and radius from config
        self.intensity = self._config.intensity
        
        # Only set radius for applicable light types
        if self._config.light_type in {"Cylinder", "Disk", "Sphere"}:
            self.radius = self._config.radius

    def _initialize(self):
        # Run super
        super()._initialize()

        # Initialize light link
        self._light_link.initialize()

    @property
    def aabb(self):
        # This is a virtual object (with no associated visual mesh), so omni returns an invalid AABB.
        # Therefore we instead return a hardcoded small value
        return th.ones(3) * -0.001, th.ones(3) * 0.001

    @property
    def light_link(self):
        """
        Returns:
            XFormPrim: Link corresponding to the light prim itself
        """
        return self._light_link

    @property
    def radius(self):
        """
        Gets this light's radius

        Returns:
            float: radius for this light
        """
        return self._light_link.get_attribute("inputs:radius")

    @radius.setter
    def radius(self, radius):
        """
        Sets this light's radius

        Args:
            radius (float): radius to set
        """
        self._light_link.set_attribute("inputs:radius", radius)

    @property
    def intensity(self):
        """
        Gets this light's intensity

        Returns:
            float: intensity for this light
        """
        return self._light_link.get_attribute("inputs:intensity")

    @intensity.setter
    def intensity(self, intensity):
        """
        Sets this light's intensity

        Args:
            intensity (float): intensity to set
        """
        self._light_link.set_attribute("inputs:intensity", intensity)

    @property
    def color(self):
        """
        Gets this light's color

        Returns:
            float: color for this light
        """
        return tuple(float(x) for x in self._light_link.get_attribute("inputs:color"))

    @color.setter
    def color(self, color):
        """
        Sets this light's color

        Args:
            color ([float, float, float]): color to set, each value in range [0, 1]
        """
        self._light_link.set_attribute("inputs:color", lazy.pxr.Gf.Vec3f(color))

    @property
    def texture_file_path(self):
        """
        Gets this light's texture file path. Only valid for dome lights.

        Returns:
            str: texture file path for this light
        """
        return str(self._light_link.get_attribute("inputs:texture:file"))

    @texture_file_path.setter
    def texture_file_path(self, texture_file_path):
        """
        Sets this light's texture file path. Only valid for dome lights.

        Args:
            texture_file_path (str): path of texture file that should be used for this light
        """
        self._light_link.set_attribute("inputs:texture:file", lazy.pxr.Sdf.AssetPath(texture_file_path))
    @property
    def light_type(self):
        """
        Returns:
            str: Type of light
        """
        return self._config.light_type
