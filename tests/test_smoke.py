"""Smoke test placeholder.

A real test suite is a TODO. For now this keeps `make test` / CI green and
catches gross import or packaging breakage (e.g. a bad dependency, a syntax
error, or import-time side effects sneaking back into the package).
"""

import burr


def test_package_imports():
    """burr imports without side effects and exposes a version string."""
    assert isinstance(burr.__version__, str)
    assert burr.__version__
