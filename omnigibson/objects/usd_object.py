import os
import tempfile

import omnigibson as og
from omnigibson.configs.base_config import USDObjectConfig
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.usd_utils import add_asset_to_stage


class USDObject(StatefulObject):
    """
    USDObjects are instantiated from a USD file. They can be composed of one
    or more links and joints. They may or may not be passive.
    """

    def __init__(self, config: USDObjectConfig):
        """
        Args:
            config (USDObjectConfig): Configuration object containing all relevant parameters for
                initializing this USD object. See the USDObjectConfig dataclass for specific parameters.
        """
        # Store config
        self._config = config
        
        # Initialize super
        super().__init__(config=config)

    def prebuild(self, stage):
        # Load the object into the given USD stage
        usd_path = self._usd_path
        if self._encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._usd_path.replace(".usd", ".encrypted.usd")
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self._usd_path), dir=og.tempdir)
            os.close(decrypted_fd)
            decrypt_file(encrypted_filename, usd_path)

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
        usd_path = self._config.usd_path
        if self._config.encrypted:
            # Create a temporary file to store the decrytped asset, load it, and then delete it
            encrypted_filename = self._config.usd_path.replace(".usd", ".encrypted.usd")
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self._config.usd_path), dir=og.tempdir)
            decrypt_file(encrypted_filename, usd_path)

        prim = add_asset_to_stage(asset_path=usd_path, prim_path=self.prim_path)

        if self._config.encrypted:
            os.close(decrypted_fd)
            # On Windows, Isaac Sim won't let go of the file until the prim is removed, so we can't delete it.
            if os.name == "posix":
                os.remove(usd_path)

        return prim

    @property
    def usd_path(self):
        """
        Returns:
            str: absolute path to this model's USD file. By default, this is the loaded usd path
                from the config
        """
        return self._config.usd_path
