from importlib import import_module

__all__ = ["T3"]


def __getattr__(name):
    if name != "T3":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".t3", __name__), name)
    globals()[name] = value
    return value
