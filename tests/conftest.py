"""Pytest configuration and fixtures for DQM-ML tests.

This module re-exports all fixtures from tests.integration.fixtures, tests.fixtures,
and tests.utils packages for convenient use in tests.
"""

import pytest

from tests.fixtures.cli_fixtures import all_classes, coco_data_dir, coco_parquet_path  # noqa: F401
from tests.fixtures.packaging_fixtures import isolated_venv, wheels_dir  # noqa: F401
from tests.fixtures.test_fixtures import mock_parquet_dataset, sample_dataframe, temp_output_path  # noqa: F401
from tests.integration.fixtures.config import tests_config  # noqa: F401
from tests.integration.fixtures.data import (  # noqa: F401
    batch_invariance_data,
    coco_csv,
    coco_data,
    coco_data_500,
    coco_data_real,
    completeness_data,
    diversity_data,
    domain_gap_bytes_data,
    full_story_data,
    large_tabular_data,
    normal_dist,
    not_normal_dist,
    not_uniform_dist,
    uniform_dist,
    visual_features_data,
)
from tests.integration.fixtures.jobs import (  # noqa: F401
    job_completeness,
    job_diversity,
    job_domain_gap,
    job_features_embeddings,
    job_representativeness,
    job_visual_features,
)
from tests.integration.fixtures.paths import output_path, test_path  # noqa: F401
from tests.integration.fixtures.pipeline_data import pipeline_data  # noqa: F401
from tests.utils.seeds import get_test_seed


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Test seed from DQM_ML_TEST_SEED env var, default 42."""
    return get_test_seed()
