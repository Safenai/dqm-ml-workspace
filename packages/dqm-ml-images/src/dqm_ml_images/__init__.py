"""DQM ML Images package for image-based data quality metrics.

This package provides image-specific metric processors for evaluating
visual data quality. It includes the VisualFeaturesProcessor which
extracts features like luminosity, contrast, blur, and entropy from images.
"""

from dqm_ml_images.visual_features import VisualFeaturesProcessor

__all__ = ["VisualFeaturesProcessor"]
