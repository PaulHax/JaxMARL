def __getattr__(name):
    if name == "make":
        from .registration import make
        return make
    if name == "registered_envs":
        from .registration import registered_envs
        return registered_envs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["make", "registered_envs"]
__version__ = "0.1.0"
