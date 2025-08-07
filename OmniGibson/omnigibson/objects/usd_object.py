import hashlib
import os
import tempfile

import omnigibson as og
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import add_asset_to_stage


# Create module logger
log = create_module_logger(module_name=__name__)


class USDObject(StatefulObject):
    """
    USDObjects are instantiated from a USD file. They can be composed of one
    or more links and joints. They may or may not be passive.
    """

    def __init__(
        self,
        name,
        usd_path,
        encrypted=False,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        expected_file_hash=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            usd_path (str): global path to the USD file to load
            encrypted (bool): whether this file is encrypted (and should therefore be decrypted) or not
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            kinematic_only (None or bool): Whether this object should be kinematic only (and not get affected by any
                collisions). If None, then this value will be set to True if @fixed_base is True and some other criteria
                are satisfied (see object_base.py post_load function), else False.
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            expected_file_hash (str): The expected hash of the file to load. This is used to check if the file has changed. None to disable check.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
                Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
                that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        self._usd_path = usd_path
        self._encrypted = encrypted
        self._expected_file_hash = expected_file_hash
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            include_default_states=include_default_states,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    def prebuild(self, stage):
        # Load the object into the given USD stage
        usd_path = self._usd_path

        if self._encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._usd_path.replace(".usd", ".encrypted.usd")
            self.check_hash(encrypted_filename)
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self._usd_path), dir=og.tempdir)
            os.close(decrypted_fd)
            decrypt_file(encrypted_filename, usd_path)
        else:
            self.check_hash(usd_path)

        # object_stage = lazy.pxr.Usd.Stage.Open(usd_path)
        # root_prim = object_stage.GetDefaultPrim()

        # The /World in the scene USD will be mapped to /World/scene_i in Isaac Sim.
        prim_path = "/World" + self._relative_prim_path

        # TODO: maybe deep copy the prim tree EXCEPT the visual meshes. How?
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = stage.DefinePrim(prim_path, "Xform")
        assert prim.GetReferences().AddReference(usd_path)

        # TODO: After we switch to a deep copy, we can enable this to remove the temporary file
        # del object_stage
        # if self._encrypted:
        #     os.remove(usd_path)

    def _load(self):
        """
        Load the object into pybullet and set it to the correct pose
        """
        usd_path = self._usd_path

        if self._encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._usd_path.replace(".usd", ".encrypted.usd")
            self.check_hash(encrypted_filename)
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self._usd_path), dir=og.tempdir)
            decrypt_file(encrypted_filename, usd_path)
        else:
            self.check_hash(usd_path)

        prim = add_asset_to_stage(asset_path=usd_path, prim_path=self.prim_path)

        if self._encrypted:
            os.close(decrypted_fd)
            # On Windows, Isaac Sim won't let go of the file until the prim is removed, so we can't delete it.
            if os.name == "posix":
                os.remove(usd_path)

        return prim

    def _post_load(self):
        super()._post_load()

        # Fix shader asset paths that may be unresolved, especially after decryption or moved assets
        root_path = os.path.dirname(self._usd_path)
        for material in self.materials:
            # If encrypted, proactively rewrite all paths relative to USD directory
            if self._encrypted:
                material.shader_update_asset_paths_with_root_path(root_path)
            # Additionally, fix any inputs that reference missing assets
            material.shader_fix_missing_asset_paths(root_path)

    def _create_prim_with_same_kwargs(self, relative_prim_path, name, load_config):
        # Add additional kwargs
        return self.__class__(
            relative_prim_path=relative_prim_path,
            usd_path=self._usd_path,
            name=name,
            category=self.category,
            scale=self.scale,
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
        )

    @property
    def usd_path(self):
        """
        Returns:
            str: absolute path to this model's USD file. By default, this is the loaded usd path
                passed in as an argument
        """
        return self._usd_path

    def check_hash(self, usd_path):
        """
        Check if the hash of the file matches the expected hash.

        Args:
            usd_path (str): The path to the USD file.

        Returns:
            bool: True if the hash matches, False otherwise.
        """
        # Hash the file to record the loaded asset's version
        hash_md5 = hashlib.md5()
        with open(usd_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()

        # If there is a file hash already in the init info, compare against it to see if the file has changed
        if self._expected_file_hash is not None:
            if file_hash != self._expected_file_hash:
                log.warn(
                    f"Object {self.name} was expected to have USD file hash {self._expected_file_hash} but loaded with {file_hash}. The saved state might be incompatible."
                )
        else:
            # If there is no expected file hash, set the expected file hash to the loaded one
            self._expected_file_hash = file_hash

            # Update the init info too so that the information gets saved with the scene.
            # TODO: Super hacky, think of a better way to preserve this info
            self._init_info["args"]["expected_file_hash"] = file_hash
