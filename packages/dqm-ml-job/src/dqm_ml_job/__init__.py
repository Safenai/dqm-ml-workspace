"""DQM ML Job package for executing data quality assessment pipelines.

This package provides the core job execution framework for running data
quality metric computations on datasets. It includes:
- CLI entry points for running jobs from YAML configurations
- Job orchestration for data loading, metric computation, and output writing
- Data loaders for various file formats (Parquet, CSV)
- Output writers for persisting results

Example:
    >>> from dqm_ml_job.cli import run
    >>> run({"config": {...}})
"""

__description__ = "DQM ML Job - Data quality assessment pipeline execution"


from dqm_ml_job.cli import run as ComputeDatasetFeatures  # noqa

__all__ = ["ComputeDatasetFeatures"]
