"""Check that pyEMA exposes exactly the sdypy-EMA code."""
import importlib

import pytest

import sdypy.EMA
import pyEMA


@pytest.mark.parametrize(
    "name", ["Model", "MAC", "MSF", "MCF", "complex_freq_to_freq_and_damp"]
)
def test_top_level_names(name):
    assert getattr(pyEMA, name) is getattr(sdypy.EMA, name)


@pytest.mark.parametrize(
    "name, target",
    [
        ("pyEMA", "EMA"),
        ("tools", "tools"),
        ("stabilization", "stabilization"),
        ("normal_modes", "normal_modes"),
        ("pole_picking", "pole_picking"),
    ],
)
def test_submodules(name, target):
    sdypy_module = importlib.import_module(f"sdypy.EMA.{target}")
    assert getattr(pyEMA, name) is sdypy_module
    assert importlib.import_module(f"pyEMA.{name}") is sdypy_module


def test_from_submodule_import():
    from pyEMA.pyEMA import Model
    from pyEMA.tools import MAC
    from pyEMA.pole_picking import SelectPoles

    assert Model is sdypy.EMA.Model
    assert MAC is sdypy.EMA.MAC
    assert SelectPoles is sdypy.EMA.pole_picking.SelectPoles
