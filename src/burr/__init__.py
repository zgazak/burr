from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-burr")
except PackageNotFoundError:
    __version__ = "unknown"

# Re-export for backward compatibility - code that calls burr.initialize_engine()
# or burr.load_environment() will still work.
from burr.bootstrap import initialize_engine, load_environment  # noqa: F401
