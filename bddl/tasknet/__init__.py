from tasknet.backend_abc import TaskNetBackend

_AVAILABLE_BACKENDS = ["iGibson"]
_backend = None


def set_backend(backend_name):
    global _backend
    if backend_name == "iGibson":
        from gibson2.task.tasknet_backend import IGibsonTaskNetBackend
        _backend = IGibsonTaskNetBackend()
    else:
        raise ValueError("Invalid backend. Currently supported backends: %s." % ", ".join(_AVAILABLE_BACKENDS))

    if not isinstance(_backend, TaskNetBackend):
        raise ValueError("Backends must implement tasknet.backend_abc.TaskNetBackend.")



def get_backend():
    if _backend is None:
        raise ValueError(
            "Before calling tasknet functions, a backend must be set using tasknet.set_backend(backend_name). "
            "Available backend names: %s." % ", ".join(_AVAILABLE_BACKENDS))

    return _backend