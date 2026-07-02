"""DQM ML Core package for data quality metrics processing.

This package provides core components for computing data quality metrics
on datasets using a streaming architecture. It includes base classes
for metric processors and implementations for common metrics like
completeness and representativeness.

Main components:
- Processor: Base class for all processors
- FeaturesProcessor: Base class for feature extraction processors
- MetricsProcessor: Base class for metric computation processors
- GapProcessor: Base class for domain-gap processors
- CompletenessProcessor: Computes data completeness scores
- RepresentativenessProcessor: Evaluates distribution representativeness
- ProcessorRunner: Orchestrator for running metrics on DataFrames
- PluginLoadedRegistry: Registry for dynamically loaded metric plugins
"""

from dqm_ml_core.api import FeaturesProcessor, GapProcessor, MetricsProcessor, Processor
from dqm_ml_core.metrics.completeness import CompletenessProcessor
from dqm_ml_core.metrics.diversity import DiversityProcessor
from dqm_ml_core.metrics.representativeness import RepresentativenessProcessor
from dqm_ml_core.utils.processor_runner import ProcessorRunner
from dqm_ml_core.utils.registry import PluginLoadedRegistry

__all__ = [
    "CompletenessProcessor",
    "DiversityProcessor",
    "FeaturesProcessor",
    "GapProcessor",
    "MetricsProcessor",
    "PluginLoadedRegistry",
    "Processor",
    "ProcessorRunner",
    "RepresentativenessProcessor",
]
