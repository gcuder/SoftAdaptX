"""Second level module import for SoftAdapt."""
from softadaptx.algorithms import *
from softadaptx.constants import *
from softadaptx.utilities import *

# adding package information and version
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "softadaptx"
__version__ = importlib_metadata.version(package_name)
