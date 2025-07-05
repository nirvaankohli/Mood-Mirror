__version__ = "0.1.0"
__author__  = "Nirvaan <nirvaan.kohli@gmail.com>"
__license__ = "MIT"

# Expose the entrypoint
from .main import main


__all__ = [
    "main",
    "__version__",
]