"""stereom: quantitative 3D structural analysis of cellular microstructures.

High-performance Python port of the MATLAB reference toolbox described in
Yang et al. / Chen et al., Acta Biomaterialia 107 (2020) 204-217 and 218-231.
See ``stereom_implementation_spec.md`` (companion document) for the full
mapping between this package and the original code.
"""

from stereom.config import StereomConfig
from stereom.volume import Volume

__all__ = ["StereomConfig", "Volume"]
__version__ = "0.1.0"
