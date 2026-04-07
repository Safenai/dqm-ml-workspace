"""Pytest configuration and fixtures for DQM-ML tests.

This module re-exports all fixtures from tests.integration.fixtures and tests.utils packages
for convenient use in tests.
"""

from tests.integration.fixtures.config import tests_config  # noqa: F401
from tests.integration.fixtures.data import (  # noqa: F401
    coco_data,
    normal_dist,
    not_normal_dist,
    not_uniform_dist,
    uniform_dist,
)
from tests.integration.fixtures.jobs import (  # noqa: F401
    job_completeness,
    job_domain_gap,
    job_representativeness,
    job_visual_features,
)
from tests.integration.fixtures.paths import output_path, test_path  # noqa: F401
