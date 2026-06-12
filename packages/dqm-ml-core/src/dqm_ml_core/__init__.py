"""DQM ML Core package for data quality metrics processing.

This package provides core components for computing data quality metrics
on datasets using a streaming architecture. It includes base classes
for metric processors and implementations for common metrics like
completeness and representativeness.

Main components:
- DatametricProcessor: Base class for all data quality metrics
- CompletenessProcessor: Computes data completeness scores
- RepresentativenessProcessor: Evaluates distribution representativeness
- MetricRunner: Orchestrator for running metrics on DataFrames
- PluginLoadedRegistry: Registry for dynamically loaded metric plugins
"""

from dqm_ml_core.api.data_processor import DatametricProcessor
from dqm_ml_core.metrics.completeness import CompletenessProcessor
from dqm_ml_core.metrics.diversity import DiversityProcessor
from dqm_ml_core.metrics.representativeness import RepresentativenessProcessor
from dqm_ml_core.utils.metric_runner import MetricRunner
from dqm_ml_core.utils.registry import PluginLoadedRegistry

__all__ = [
    "CompletenessProcessor",
    "DatametricProcessor",
    "DiversityProcessor",
    "MetricRunner",
    "PluginLoadedRegistry",
    "RepresentativenessProcessor",
]
