"""Job fixtures for DQM-ML tests.

This module provides fixtures for generating test job configurations.
"""

from pathlib import Path
from typing import Any

import pytest
from tests.utils.jobs import generate_job


@pytest.fixture(scope="session")
def job_representativeness(
    test_path: str,
    normal_dist: Any,
    not_normal_dist: Any,
    uniform_dist: Any,
    not_uniform_dist: Any,
) -> None:
    """Generate test job configurations for representativeness tests.

    Args:
        test_path: Path to the tests directory.
        normal_dist: Fixture providing normal distribution data.
        not_normal_dist: Fixture providing non-normal distribution data.
        uniform_dist: Fixture providing uniform distribution data.
        not_uniform_dist: Fixture providing non-uniform distribution data.
    """
    test_list = [
        {
            "test_name": "normal_distribution",
            "parquet": "normal_distribution.parquet",
        },
        {
            "test_name": "not_normal_distribution",
            "parquet": "not_normal_distribution.parquet",
        },
        {
            "test_name": "uniform_distribution",
            "parquet": "uniform_distribution.parquet",
        },
        {
            "test_name": "not_uniform_distribution",
            "parquet": "not_uniform_distribution.parquet",
        },
        {"test_name": "batch", "parquet": "normal_distribution.parquet"},
    ]

    generate_job(
        processor_name="representativeness",
        parquets_path=Path(test_path) / "outputs/data",
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_completeness(test_path: str) -> None:
    """Generate test job configurations for completeness tests.

    Args:
        test_path: Path to the tests directory.
    """
    test_list = [
        {"test_name": "completeness", "parquet": "completeness.parquet"},
        {"test_name": "completeness_batch", "parquet": "completeness.parquet"},
    ]

    generate_job(
        processor_name="completeness",
        parquets_path=Path(test_path) / "data",
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_domain_gap(test_path: str) -> None:
    """Generate test job configurations for domain gap tests.

    Args:
        test_path: Path to the tests directory.
    """
    gen_path = Path(test_path) / "outputs/data"
    metrics = ["fid", "klmvn_diag", "mmd_linear", "wasserstein_1d", "mmd_rbf", "mmd_poly", "pad", "cmd"]

    for metric in metrics:
        generate_job(
            processor_name="domain_gap",
            parquets_path=gen_path,
            test_list=[{"test_name": "", "parquet": "target_1000.parquet"}],
            output_category="delta_metrics",
            test_path=test_path,
            metric_name=metric,
            parquet_source_path=Path(gen_path) / "source_1000.parquet",
        )

    generate_job(
        processor_name="domain_gap",
        parquets_path=Path(test_path) / "data",
        test_list=[
            {
                "test_name": "wasserstein_bytes",
                "parquet": "target_bytes.parquet",
            }
        ],
        output_category="delta_metrics",
        test_path=test_path,
        metric_name="wasserstein_1d",
        parquet_source_path=Path(test_path) / "data/source_bytes.parquet",
    )


@pytest.fixture(scope="session")
def job_diversity(test_path: str, diversity_data: Any) -> None:
    """Generate test job configurations for diversity tests.

    Args:
        test_path: Path to the tests directory.
        diversity_data: Fixture that generates diversity test parquet.
    """
    test_list = [
        {"test_name": "diversity", "parquet": "diversity.parquet"},
        {"test_name": "diversity_batch", "parquet": "diversity.parquet"},
    ]

    generate_job(
        processor_name="diversity",
        parquets_path=Path(test_path) / "outputs/data",
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_visual_features(test_path: str) -> None:
    """Generate test job configurations for visual features tests.

    Args:
        test_path: Path to the tests directory.
    """
    test_list = [
        {"test_name": "visual_features", "parquet": "visual_features.parquet"},
        {
            "test_name": "visual_features_batch",
            "parquet": "visual_features.parquet",
        },
        {
            "test_name": "visual_features_path",
            "parquet": "visual_features_path.parquet",
        },
    ]

    generate_job(
        processor_name="visual_features",
        parquets_path=Path(test_path) / "data",
        test_list=test_list,
        output_category="features",
        test_path=test_path,
    )
