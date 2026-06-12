"""Pytest configuration and fixtures for DQM-ML tests.

This module re-exports all fixtures from tests.integration.fixtures, tests.fixtures,
and tests.utils packages for convenient use in tests.
"""

from tests.fixtures.cli_fixtures import (  # noqa: F401
    all_classes,
    coco_data_dir,
    coco_parquet_path,
    fixtures_dir,
    split_by_config_path,
)
from tests.fixtures.test_fixtures import mock_parquet_dataset, sample_dataframe, temp_output_path  # noqa: F401
from tests.integration.fixtures.config import tests_config  # noqa: F401
from tests.integration.fixtures.data import (  # noqa: F401
    coco_data,
    coco_data_500,
    diversity_data,
    normal_dist,
    not_normal_dist,
    not_uniform_dist,
    uniform_dist,
)
from tests.integration.fixtures.jobs import (  # noqa: F401
    job_completeness,
    job_diversity,
    job_domain_gap,
    job_representativeness,
    job_visual_features,
)
from tests.integration.fixtures.paths import output_path, test_path  # noqa: F401
