"""Load a drill module from mine/ (default) or solutions/ (DRILL_IMPL=solutions)."""
import importlib
import os


def load(name: str):
    impl = os.environ.get("DRILL_IMPL", "mine")
    if impl not in ("mine", "solutions"):
        raise ValueError(f"DRILL_IMPL must be 'mine' or 'solutions', got {impl!r}")
    return importlib.import_module(f"{impl}.{name}")
