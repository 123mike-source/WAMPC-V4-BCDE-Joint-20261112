"""Reuse the unchanged timed action/data contracts without legacy package init."""
import importlib
from pathlib import Path
import sys
import types


def module(suffix):
    name = 'method.irwm_bev_t_large._timed_source'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'irwm_bev_t')]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(name + '.' + suffix)
