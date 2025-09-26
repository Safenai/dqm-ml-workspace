"""
Data processors module.

This module contains classes for processing data and computing metrics.
"""

from dqm_ml_core.completeness import CompletenessProcessor
#from dqm_ml_pipeline.core.metrics.domain_gap import DomainGapProcessor, ImageEmbeddingProcessor

# Import DQM-ML metrics
from  dqm_ml_core.representativeness import RepresentativenessProcessor
#from  dqm_ml_pipeline.core.visual_features import VisualFeaturesProcessor


from .image.feature import ImageFeaturesProcessor
from .image.loader import ImageLoaderProcessor

# Registry of supported data loaders
dqml_metrics_registry = {
    "load_image": ImageLoaderProcessor,
#    "visual_features": VisualFeaturesProcessor,
    "image_features": ImageFeaturesProcessor,
    "representativeness": RepresentativenessProcessor,
#   "domain_gap": DomainGapProcessor,           
#    "image_embedding": ImageEmbeddingProcessor,
    "completeness": CompletenessProcessor,
}

__all__ = [
    "dqml_metrics_registry"
]
