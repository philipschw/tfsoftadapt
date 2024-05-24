"""Second level module import for SoftAdapt."""
from .algorithms import *
from .constants import *
from .utilities import *

# adding package information and version
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "tfsoftadapt"
__version__ = importlib_metadata.version(package_name)
