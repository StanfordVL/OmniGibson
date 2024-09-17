import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty


class ClothStateMixin(BaseObjectState):
    """
    This class is a subclass of BaseObjectState that adds dependencies assuming the owned object is PrimType.CLOTH
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Only compatible with cloth objects
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check for cloth type
        if obj.prim_type != PrimType.CLOTH:
            return (
                False,
                f"Cannot use ClothStateMixin {cls.__name__} with rigid object, make sure object is created "
                f"with prim_type=PrimType.CLOTH!",
            )

        # Check for GPU pipeline
        if og.sim.device == "cpu":
            return False, f"Must be using GPU dynamics in order to use object state {cls.__name__}."

        return True, None

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ClothStateMixin")
        return classes
