from behavior.backend_abc import BEHAVIORBackend

_AVAILABLE_BACKENDS = ["iGibson"]
_backend = None


def set_backend(backend_name):
    global _backend
    if backend_name == "iGibson":
        from igibson.task.behavior_backend import IGibsonBEHAVIORBackend
        _backend = IGibsonBEHAVIORBackend()
    else:
        raise ValueError("Invalid backend. Currently supported backends: %s." % ", ".join(_AVAILABLE_BACKENDS))

    if not isinstance(_backend, BEHAVIORBackend):
        raise ValueError("Backends must implement behavior.backend_abc.BEHAVIORBackend.")



def get_backend():
    if _backend is None:
        raise ValueError(
            "Before calling behavior functions, a backend must be set using behavior.set_backend(backend_name). "
            "Available backend names: %s." % ", ".join(_AVAILABLE_BACKENDS))

    return _backend