from importlib.metadata import PackageNotFoundError, version

from .easyocr import Reader

__all__ = ["Reader", "__version__"]

try:
    __version__ = version("easyocr")
except PackageNotFoundError:  # running from a source checkout without metadata
    __version__ = "1.7.3"
