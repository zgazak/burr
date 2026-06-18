"""
Sky coverage task source module.

This module provides systematic sky coverage using HEALPix tessellation
and traveling salesman optimization for efficient telescope scheduling.
"""

from .manager import SkyCoverageManager, SkyPointing, SparseMap, TravelingSalesmanOptimizer

__all__ = [
    "SkyCoverageManager",
    "SkyPointing",
    "SparseMap",
    "TravelingSalesmanOptimizer",
]
