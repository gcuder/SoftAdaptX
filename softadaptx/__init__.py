"""Second level module import for softadaptx."""

from softadaptx.algorithms import LossWeightedSoftAdapt, NormalizedSoftAdapt, SoftAdapt

__all__ = ["LossWeightedSoftAdapt", "NormalizedSoftAdapt", "SoftAdapt"]

# adding package information and version
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata as importlib_metadata_fallback

    importlib_metadata = importlib_metadata_fallback

package_name = "softadaptx"
__version__ = importlib_metadata.version(package_name)
