"""DQM ML v2 package for data quality assessment.

This is the main entry point for the DQM-ML v2 CLI. It provides commands
for processing data quality metrics and listing available plugins.

Commands:
    version: Display version information
    list: List available metrics, data loaders, and output writers
    process: Run a data quality assessment job
"""

from dqm_ml.__main__ import execute

__all__ = ["execute"]
