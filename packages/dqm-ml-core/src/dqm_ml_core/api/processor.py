"""Base processor class.

This module contains the Processor base class that all
metrics, features, and gap processors must inherit from.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Processor:
    """
    Base class for all Data Quality metrics, feature extractors, and gap processors.

    Provides shared initialization, failure-rate checking, and column resolution.
    Lifecycle methods are defined in the appropriate subclass:
    :class:`FeaturesProcessor`, :class:`MetricsProcessor`, or :class:`GapProcessor`.
    """

    def __init__(self, name: str, config: dict[str, Any] | None):
        """
        Initialize the processor.

        Args:
            name: Unique name of the processor instance.
            config: Configuration dictionary (optional).
        """
        self.name = name
        self.config = config or {}
        self.errors_config = None
        self.compute_device: str = "cpu"
        self.compute_seed: int | None = None
        self.current_path_prefix: dict[str, str] = {}
        self._failure_count = 0
        self._total_count = 0

        self.input_columns: list[str] = []
        self.exclude_columns: list[str] | None = None
        self.columns_config = None
        if "columns" in self.config and isinstance(self.config["columns"], dict):
            from dqm_ml_core.models.columns import ColumnsConfig

            self.columns_config = ColumnsConfig.model_validate(self.config["columns"])
            self.input_columns = self.columns_config.input
            self.exclude_columns = self.columns_config.exclude

    def _check_failure_rate(self) -> None:
        if self.errors_config and self.errors_config.max_failure_rate is not None and self._total_count > 0:
            failure_rate = self._failure_count / self._total_count
            if failure_rate > self.errors_config.max_failure_rate:
                raise RuntimeError(
                    f"Failure rate {failure_rate:.1%} exceeds max {self.errors_config.max_failure_rate:.1%}"
                )

    def needed_columns(self) -> list[str]:
        """
        Return the list of raw input columns required for processing.

        Returns:
            A list of column names.
        """
        return self.input_columns

    def reset(self) -> None:
        """Reset per-selection state between dataselections.

        Processors that cache per-selection state (e.g. histogram bin
        edges in RepresentativenessProcessor) MUST override this to
        clear that state.  Called by DatasetJob after each selection.
        """
