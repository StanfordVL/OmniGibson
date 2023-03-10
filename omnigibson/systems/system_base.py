import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.python_utils import classproperty, assert_valid_key, \
    SerializableNonInstance, UniquelyNamedNonInstance
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Global dicts that will contain mappings of all the systems
REGISTERED_SYSTEMS = dict()


def get_system(system_name):
    system = REGISTERED_SYSTEMS[system_name]
    # If the scene has already been loaded when get_system is called, add the system to the registry and initialize
    if og.sim and og.sim.scene and og.sim.scene.loaded and og.sim.scene.system_registry("name", system_name) is None:
        og.sim.scene.system_registry.add(obj=system)
        system.initialize()
    return system


class BaseSystem(SerializableNonInstance, UniquelyNamedNonInstance):
    """
    Base class for all systems. These are non-instance objects that should be used globally for a given environment.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc.
    """
    def __init_subclass__(cls, **kwargs):
        # Run super init
        super().__init_subclass__(**kwargs)

        global REGISTERED_SYSTEMS
        # Register this system if requested
        if cls._register_system:
            REGISTERED_SYSTEMS[cls.__name__] = cls

        cls._uuid = abs(hash(cls.__name__)) % (10 ** 8)

    initialized = False
    _uuid = None

    @classproperty
    def name(cls):
        # Class name is the unique name assigned
        return cls.__name__

    @classproperty
    def uuid(cls):
        return cls._uuid

    @classproperty
    def prim_path(cls):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        return f"/World/{cls.name}"

    @classproperty
    def _register_system(cls):
        """
        Returns:
            bool: True if this system should be registered (i.e.: it is not an intermediate class but a "final" subclass
                representing a system we'd actually like to use, e.g.: water, dust, etc. Should be set by the subclass
        """
        # We assume we aren't registering by default
        return False

    @classmethod
    def initialize(cls):
        """
        Initializes this system
        """
        assert not cls.initialized, f"Already initialized system {cls.name}!"
        og.sim.stage.DefinePrim(cls.prim_path, "Scope")
        cls.initialized = True

    @classmethod
    def clear(cls):
        """
        Clears this system, so that it may possibly be re-initialized. Useful for, e.g., when loading from a new
        scene during the same sim instance
        """
        if cls.initialized:
            cls.reset()
            cls.initialized = False

    @classmethod
    def reset(cls):
        """
        Reset this system
        """
        pass

    @classmethod
    def get_active_systems(cls):
        """
        Returns:
            dict: Mapping from system name to system for all systems that are subclasses of this system AND active (initialized)
        """
        return {system.name: system for system in REGISTERED_SYSTEMS.values() if issubclass(system, cls) and system.initialized}

    @classmethod
    def get_systems(cls):
        """
        Returns:
            dict: Mapping from system name to system for all systems that are subclasses of this system
        """
        return {system.name: system for system in REGISTERED_SYSTEMS.values() if issubclass(system, cls)}

    def __init__(self):
        raise ValueError("System classes should not be created!")
