__version__ = "0.26.1"

# from .pyEMA import Model
# from .tools import *
# from . import stabilization
# from . import normal_modes
# from . import pole_picking

# Importing code from SDyPy-EMA package where it is maintained
from sdypy.EMA import Model
from sdypy.EMA.tools import *
from sdypy.EMA import stabilization
from sdypy.EMA import normal_modes
from sdypy.EMA import pole_picking

# pyEMA moving to SDyPy warning
import warnings
warnings.warn('The imported code is channeled from the SDyPy package. It is recommended to use the SDyPy package instead of pyEMA. To use the latest code from SDyPy: `pip install sdypy` and `from sdypy import EMA`.', DeprecationWarning)