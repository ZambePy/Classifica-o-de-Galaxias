"""GALAXIA - classificador hierarquico de objetos astronomicos."""

__version__ = "0.1.0"

from astro_classifier.taxonomy import (
    GALAXY_CLASSES,
    NEBULA_CLASSES,
    OBJECT_CLASSES,
    Level,
)

__all__ = ["Level", "OBJECT_CLASSES", "GALAXY_CLASSES", "NEBULA_CLASSES", "__version__"]
