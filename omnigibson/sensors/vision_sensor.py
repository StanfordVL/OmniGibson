import numpy as np
import time
import gym

import omnigibson as og
from omnigibson.sensors.sensor_base import BaseSensor
from omnigibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, MAX_VIEWER_SIZE, VALID_OMNI_CHARS
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sim_utils import set_carb_setting
from omnigibson.utils.ui_utils import dock_window, suppress_omni_log
from omnigibson.utils.usd_utils import get_camera_params
from omnigibson.utils.transform_utils import euler2quat, quat2euler

import omni.ui
import omni.replicator.core as rep
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdRender
from omni.kit.viewport.window import get_viewport_window_instances
from omni.kit.viewport.utility import create_viewport_window

# Make sure synthetic data extension is enabled
ext_manager = og.app.app.get_extension_manager()
ext_manager.set_extension_enabled("omni.syntheticdata", True)


# Duplicate of simulator's render method, used so that this can be done before simulator is created!
def render():
    """
    Refreshes the Isaac Sim app rendering components including UI elements and view ports..etc.
    """
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", False)
    og.app.update()
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", True)


class VisionSensor(BaseSensor):
    """
    Vision sensor that handles a variety of modalities, including:

        - RGB (normal)
        - Depth (normal, linear)
        - Normals
        - Segmentation (semantic, instance)
        - Optical flow
        - 2D Bounding boxes (tight, loose)
        - 3D Bounding boxes
        - Camera state

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Default is "all", which corresponds
            to all modalities being used. Otherwise, valid options should be part of cls.all_modalities.
            For this vision sensor, this includes any of:
                {rgb, depth, depth_linear, normal, seg_semantic, seg_instance, flow, bbox_2d_tight,
                bbox_2d_loose, bbox_3d, camera}
        enabled (bool): Whether this sensor should be enabled by default
        noise (None or BaseSensorNoise): If specified, sensor noise model to apply to this sensor.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this sensor's prim at runtime.
        image_height (int): Height of generated images, in pixels
        image_width (int): Width of generated images, in pixels
        viewport_name (None or str): If specified, will link this camera to the specified viewport, overriding its
            current camera. Otherwise, creates a new viewport
    """
    # Define raw sensor types
    _RAW_SENSOR_TYPES = dict(
        rgb="rgb",
        depth="distance_to_camera",
        depth_linear="distance_to_image_plane",
        normal="normals",
        seg_semantic="semantic_segmentation",
        seg_instance="instance_segmentation",
        seg_instance_id="instance_id_segmentation",
        flow="motion_vectors",
        bbox_2d_tight="bounding_box_2d_tight",
        bbox_2d_loose="bounding_box_2d_loose",
        bbox_3d="bounding_box_3d",
        camera_params="camera_params",
    )

    # Persistent dictionary of sensors, mapped from prim_path to sensor
    SENSORS = dict()

    def __init__(
        self,
        prim_path,
        name,
        modalities="all",
        enabled=True,
        noise=None,
        load_config=None,
        image_height=128,
        image_width=128,
        viewport_name=None,
    ):
        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["image_height"] = image_height
        load_config["image_width"] = image_width
        load_config["viewport_name"] = viewport_name

        # Create variables that will be filled in later at runtime
        self._sd = None             # synthetic data interface
        self._viewport = None       # Viewport from which to grab data
        self._annotators = None     # Replicator backend nodes that generate the raw data
        self._render_product_path = None

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            modalities=modalities,
            enabled=enabled,
            noise=noise,
            load_config=load_config,
        )

    def _load(self):
        # Define a new camera prim at the current stage
        # Note that we can't use og.sim.stage here because the vision sensors get loaded first
        return UsdGeom.Camera.Define(get_current_stage(), self._prim_path).GetPrim()

    def _post_load(self):
        # run super first
        super()._post_load()

        # Add this sensor to the list of global sensors
        self.SENSORS[self._prim_path] = self

        # Create the camera backend prim for low-level APIs to actually generate data
        resolution = (self._load_config["image_width"], self._load_config["image_height"])
        self._render_product_path = rep.create.render_product(self._prim_path, resolution=resolution)

        # Create a new viewport to link to this camera or link to a pre-existing one
        viewport_name = self._load_config["viewport_name"]
        if viewport_name is not None:
            vp_names_to_handles = {vp.name: vp for vp in get_viewport_window_instances()}
            assert_valid_key(key=viewport_name, valid_keys=vp_names_to_handles, name="viewport name")
            viewport = vp_names_to_handles[viewport_name]
        else:
            viewport = create_viewport_window()
            # Take a render step to make sure the viewport is generated before docking it
            render()
            # Grab the newly created viewport and dock it to the GUI
            # The first viewport is always the "main" global camera, and any additional cameras are auxiliary views
            # These auxiliary views will be stacked in a single column
            # Thus, the first auxiliary viewport should be generated to the left of the main dockspace, and any
            # subsequent viewports should be equally spaced according to the number of pre-existing auxiliary views
            n_auxiliary_sensors = len(self.SENSORS) - 1
            if n_auxiliary_sensors == 1:
                # This is the first auxiliary viewport, dock to the left of the main dockspace
                dock_window(space=omni.ui.Workspace.get_window("DockSpace"), name=viewport.name,
                            location=omni.ui.DockPosition.LEFT, ratio=0.25)
            elif n_auxiliary_sensors > 1:
                # This is any additional auxiliary viewports, dock equally-spaced in the auxiliary column
                # We also need to re-dock any prior viewports!
                for i in range(2, n_auxiliary_sensors + 1):
                    dock_window(space=omni.ui.Workspace.get_window(f"Viewport {i - 1}"), name=f"Viewport {i}",
                                location=omni.ui.DockPosition.BOTTOM, ratio=(1 + n_auxiliary_sensors - i) / (2 + n_auxiliary_sensors - i))

        self._viewport = viewport

        # Link the camera and viewport together
        self._viewport.viewport_api.set_active_camera(self._prim_path)

        # Requires 3 render updates to propagate changes
        for i in range(3):
            render()

        # Set the viewer and rendering size (requires taking one render step afterwards)
        self._viewport.viewport_api.set_texture_resolution(resolution)
        UsdRender.Product(get_prim_at_path(self._render_product_path)).GetResolutionAttr().Set(Gf.Vec2i(*resolution))

        # Requires 3 render updates to propagate changes
        for i in range(3):
            render()

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize sensors
        self._annotators = {modality: None for modality in self.all_modalities}
        self.initialize_sensors(names=self.ordered_modalities)

    def initialize_sensors(self, names):
        """Initializes a raw sensor in the simulation.

        Args:
            names (str or list of str): Name of the raw sensor(s) to initialize.
                If they are not part of self._RAW_SENSOR_TYPES' keys, we will simply pass over them
        """
        names = {names} if isinstance(names, str) else set(names)
        for name in names:
            self._add_modality_to_backend(modality=name)

        # Needs 2 render steps, with some added time in between to force render buffers to populate
        # Synthetic data also occasionally outputs spurious warnings here so we suppress them
        with suppress_omni_log(channels=["omni.syntheticdata.plugin"]):
            render()
            time.sleep(0.05)
            render()

    def _get_obs(self):
        # Make sure we're initialized
        assert self.initialized, "Cannot grab vision observations without first initializing this VisionSensor!"

        # Run super first to grab any upstream obs
        obs = super()._get_obs()

        # Process each sensor modality individually
        for modality in self.ordered_modalities:
            raw_obs = self._annotators[modality].get_data()
            # Obs is either a dictionary of {"data":, ..., "info": ...} or a direct array
            obs[modality] = raw_obs["data"] if isinstance(raw_obs, dict) else raw_obs

        return obs

    def add_modality(self, modality):
        # Check if we already have this modality (if so, no need to initialize it explicitly)
        should_initialize = modality not in self.modalities

        # Run super
        super().add_modality(modality=modality)

        # We also need to initialize this new modality
        if should_initialize:
            self.initialize_sensors(names=modality)

    def remove_modality(self, modality):
        # Check if we don't have this modality (if not, no need to remove it explicitly)
        should_remove = modality in self.modalities

        # Run super
        super().remove_modality(modality=modality)

        # We also need to initialize this new modality
        if should_remove:
            self._remove_modality_from_backend(modality=modality)

    def get_local_pose(self):
        # We have to overwrite this because camera prims can't set their quat for some reason ):
        xform_translate_op = self.get_attribute("xformOp:translate")
        xform_orient_op = self.get_attribute("xformOp:rotateXYZ")
        return np.array(xform_translate_op), euler2quat(np.array(xform_orient_op))

    def remove(self):
        # Remove from global sensors dictionary
        self.SENSORS.pop(self._prim_path)

        # Remove viewport
        self._viewport.destroy()

        # Run super
        super().remove()

    def _add_modality_to_backend(self, modality):
        """
        Helper function to add specified modality @modality to the omniverse Replicator backend so that its data is
        generated during get_obs()

        Args:
            modality (str): Name of the modality to add to the Replicator backend
        """
        if self._annotators.get(modality, None) is None:
            self._annotators[modality] = rep.AnnotatorRegistry.get_annotator(self._RAW_SENSOR_TYPES[modality])
            self._annotators[modality].attach([self._render_product_path])

    def _remove_modality_from_backend(self, modality):
        """
        Helper function to remove specified modality @modality from the omniverse Replicator backend so that its data is
        no longer generated during get_obs()

        Args:
            modality (str): Name of the modality to remove from the Replicator backend
        """
        if self._annotators.get(modality, None) is not None:
            self._annotators[modality].detach([self._render_product_path])
            self._annotators[modality] = None

    @property
    def viewer_visibility(self):
        """
        Returns:
            bool: Whether the viewer is visible or not
        """
        return self._viewport.visible

    @viewer_visibility.setter
    def viewer_visibility(self, visible):
        """
        Sets whether the viewer should be visible or not in the Omni UI

        Args:
            visible (bool): Whether the viewer should be visible or not
        """
        self._viewport.visible = visible
        # Requires 1 render update to propagate changes
        render()

    @property
    def image_height(self):
        """
        Returns:
            int: Image height of this sensor, in pixels
        """
        return self._viewport.viewport_api.get_texture_resolution()[1]

    @image_height.setter
    def image_height(self, height):
        """
        Sets the image height @height for this sensor

        Args:
            height (int): Image height of this sensor, in pixels
        """
        width, _ = self._viewport.viewport_api.get_texture_resolution()
        self._viewport.viewport_api.set_texture_resolution((width, height))
        UsdRender.Product(get_prim_at_path(self._render_product_path)).GetResolutionAttr().Set(Gf.Vec2i(width, height))
        # Requires 3 updates to propagate changes
        for i in range(3):
            render()

    @property
    def image_width(self):
        """
        Returns:
            int: Image width of this sensor, in pixels
        """
        return self._viewport.viewport_api.get_texture_resolution()[0]

    @image_width.setter
    def image_width(self, width):
        """
        Sets the image width @width for this sensor

        Args:
            width (int): Image width of this sensor, in pixels
        """
        _, height = self._viewport.viewport_api.get_texture_resolution()
        self._viewport.viewport_api.set_texture_resolution((width, height))
        UsdRender.Product(get_prim_at_path(self._render_product_path)).GetResolutionAttr().Set(Gf.Vec2i(width, height))
        # Requires 3 updates to propagate changes
        for i in range(3):
            render()

    @property
    def clipping_range(self):
        """
        Returns:
            2-tuple: [min, max] value of the sensor's clipping range, in meters
        """
        return np.array(self.get_attribute("clippingRange"))

    @clipping_range.setter
    def clipping_range(self, limits):
        """
        Sets the clipping range @limits for this sensor

        Args:
            limits (2-tuple): [min, max] value of the sensor's clipping range, in meters
        """
        self.set_attribute(attr="clippingRange", val=Gf.Vec2f(*limits))
        # In order for sensor changes to propagate, we must toggle its visibility
        self.visible = False
        # A single update step has to happen here before we toggle visibility for changes to propagate
        render()
        self.visible = True

    @property
    def focal_length(self):
        """
        Returns:
            float: focal length of this sensor, in meters
        """
        return self.get_attribute("focalLength")

    @focal_length.setter
    def focal_length(self, length):
        """
        Sets the focal length @length for this sensor

        Args:
            length (float): focal length of this sensor, in meters
        """
        self.set_attribute("focalLength", length)

    @property
    def camera_parameters(self):
        """
        Returns:
            dict: Keyword-mapped relevant instrinsic and extrinsic camera parameters for this vision sensor
        """
        # Add the camera params modality if it doesn't already exist
        if "camera_params" not in self._annotators:
            self.initialize_sensors(names="camera_params")

        # Grab and return the parameters
        return self._annotators["camera_params"].get_data()

    @property
    def _obs_space_mapping(self):
        # Generate the complex space types for special modalities:
        # {"bbox_2d_tight", "bbox_2d_loose", "bbox_3d"}
        bbox_3d_space = gym.spaces.Sequence(space=gym.spaces.Tuple((
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=int),  # semanticId
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # x_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # y_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # z_min
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # x_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # y_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float), # z_max
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=float), # transform
            gym.spaces.Box(low=0, high=1.0, shape=(), dtype=float),  # occlusion ratio
        )))

        bbox_2d_space = gym.spaces.Sequence(space=gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=int),  # semanticId
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=int),  # x_min
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=int),  # y_min
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=int),  # x_max
            gym.spaces.Box(low=0, high=MAX_VIEWER_SIZE, shape=(), dtype=int),  # y_max
            gym.spaces.Box(low=0, high=1.0, shape=(), dtype=float),  # occlusion ratio
        )))

        obs_space_mapping = dict(
            rgb=((self.image_height, self.image_width, 4), 0, 255, np.uint8),
            depth=((self.image_height, self.image_width), 0.0, np.inf, np.float32),
            depth_linear=((self.image_height, self.image_width), 0.0, np.inf, np.float32),
            normal=((self.image_height, self.image_width, 4), -1.0, 1.0, np.float32),
            seg_semantic=((self.image_height, self.image_width), 0, MAX_CLASS_COUNT, np.uint32),
            seg_instance=((self.image_height, self.image_width), 0, MAX_INSTANCE_COUNT, np.uint32),
            seg_instance_id=((self.image_height, self.image_width), 0, MAX_INSTANCE_COUNT, np.uint32),
            flow=((self.image_height, self.image_width, 4), -np.inf, np.inf, np.float32),
            bbox_2d_tight=bbox_2d_space,
            bbox_2d_loose=bbox_2d_space,
            bbox_3d=bbox_3d_space,
        )

        return obs_space_mapping

    @classmethod
    def clear(cls):
        """
        Clears all cached sensors that have been generated. Should be used when the simulator is completely reset; i.e.:
        all objects on the stage are destroyed
        """
        for sensor in cls.SENSORS.values():
            # Destroy any sensor that is not attached to the main viewport window
            if sensor._viewport.name != "Viewport":
                sensor._viewport.destroy()

        # Render to update
        render()

        cls.SENSORS = dict()

    @classproperty
    def all_modalities(cls):
        # All native sensors count as valid obs modalities EXCEPT camera params since (a) its data type is not uniform,
        # and (b) it is accessed directly as a property of this VisionSensor
        return {k for k in cls._RAW_SENSOR_TYPES.keys() if k != "camera_params"}

    @classproperty
    def no_noise_modalities(cls):
        # bounding boxes should not have noise
        return {"bbox_2d_tight", "bbox_2d_loose", "bbox_3d"}
