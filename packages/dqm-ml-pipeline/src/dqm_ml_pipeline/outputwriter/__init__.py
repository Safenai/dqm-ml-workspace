"""
Data processors module.

This module contains classes for processing data and computing metrics.
"""

from dqm_ml_pipeline.outputwriter.parquet import ParquetOutputWriter

dqml_outputs_registry = {"parquet": ParquetOutputWriter}


__all__ = ["ParquetOutputWriter", "dqml_outputs_registry"]
