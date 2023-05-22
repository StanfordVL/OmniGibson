"""
This file contains the base wrapper class for OmnOmniGibson environments
Wrappers are useful for data collection and logging. Highly recommended.
"""


class BaseWrapper:
    """
    Base class for all wrappers in OmniGibson

    Args:
        obj (any): Arbitrary python object instance to wrap
    """

    def __init__(self, obj):
        # Set the internal attributes -- store wrapped obj
        self.wrapped_obj = obj

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        obj = self.wrapped_obj
        while True:
            if isinstance(obj, BaseWrapper):
                if obj.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                obj = obj.wrapped_obj
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped object

        Returns:
            any: The unwrapped object instance
        """
        return self.wrapped_obj.unwrapped if hasattr(self.wrapped_obj, "unwrapped") else self.wrapped_obj

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.wrapped_obj, attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_obj:
                    return self
                return result
            return hooked
        else:
            return orig_attr
