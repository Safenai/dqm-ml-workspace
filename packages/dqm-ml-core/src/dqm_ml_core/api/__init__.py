"""API modules for DQM ML Core.

This package contains the base API components for data metric processors,
including the DatametricProcessor base class that all metric processors
must inherit from.
"""

from dqm_ml_core.api.data_processor import DatametricProcessor

__all__ = ["DatametricProcessor"]
