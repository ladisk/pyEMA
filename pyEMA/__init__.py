"""Experimental and operational modal analysis.

pyEMA is a port of sdypy-EMA (https://github.com/sdypy/sdypy-EMA), where the
code is maintained. This package re-exports sdypy-EMA under the pyEMA name.
"""
__version__ = "0.26.1"

import sys as _sys

from sdypy.EMA import *
from sdypy.EMA import EMA as pyEMA
from sdypy.EMA import tools, stabilization, normal_modes, pole_picking

# Register the submodules under the pyEMA name, so that e.g. `import pyEMA.tools`
# returns sdypy.EMA.tools instead of looking for a file in this package.
for _name in ("pyEMA", "tools", "stabilization", "normal_modes", "pole_picking"):
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]
del _name

# pyEMA moving to SDyPy warning
import warnings
warnings.warn('The imported code is channeled from the SDyPy package. It is recommended to use the SDyPy package instead of pyEMA. To use the latest code from SDyPy: `pip install sdypy` and `from sdypy import EMA`.', DeprecationWarning)