__version__ = "0.25"

from .pyEMA import Model
from .tools import *

from . import stabilization
from . import normal_modes
from . import pole_picking

# pyEMA moving to SDyPy warning
import warnings
warnings.warn('This is the last version of pyEMA and will not longer be maintained since it is moving to the SDyPy package. To use the latest code from SDyPy: `pip install sdypy` and `from sdypy import EMA`.')