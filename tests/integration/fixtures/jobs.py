"""Job fixtures for DQM-ML tests.

This module provides fixtures for generating test job configurations.
"""

from pathlib import Path
from typing import Any

import pytest
from tests.utils.jobs import generate_job

_OUTPUT_DATA_DIR = "outputs/data"
_NORMAL_DIST_PARQUET = "normal_distribution.parquet"
_COMPLETENESS_PARQUET = "completeness.parquet"
_DIVERSITY_PARQUET = "diversity.parquet"
_VISUAL_FEATURES_PARQUET = "visual_features.parquet"


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
            "parquet": _NORMAL_DIST_PARQUET,
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
        {"test_name": "batch", "parquet": _NORMAL_DIST_PARQUET},
        {
            "test_name": "normal_distribution_custom_interpretations",
            "parquet": _NORMAL_DIST_PARQUET,
        },
        {
            "test_name": "normal_distribution_shannon_threshold",
            "parquet": _NORMAL_DIST_PARQUET,
        },
    ]

    generate_job(
        processor_name="representativeness",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_completeness(test_path: str, completeness_data: Any) -> None:
    """Generate test job configurations for completeness tests.

    Args:
        test_path: Path to the tests directory.
        completeness_data: Fixture that generates completeness.parquet.
    """
    test_list = [
        {"test_name": "completeness", "parquet": _COMPLETENESS_PARQUET},
        {"test_name": "completeness_batch", "parquet": _COMPLETENESS_PARQUET},
        {"test_name": "completeness_no_per_column", "parquet": _COMPLETENESS_PARQUET},
        {"test_name": "completeness_no_overall", "parquet": _COMPLETENESS_PARQUET},
    ]

    generate_job(
        processor_name="completeness",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_domain_gap(test_path: str, domain_gap_bytes_data: Any) -> None:
    """Generate test job configurations for domain gap tests.

    Args:
        test_path: Path to the tests directory.
        domain_gap_bytes_data: Fixture that generates bytes parquets.
    """
    gen_path = Path(test_path) / _OUTPUT_DATA_DIR
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

    # Variant configs for testing summary and distance parameters
    variants = [
        "fid_no_sum_outer",
        "mmd_rbf_no_store",
        "pad_mae",
        "mmd_rbf_gamma2",
        "wasserstein_custom_hist",
    ]
    for variant in variants:
        generate_job(
            processor_name="domain_gap",
            parquets_path=gen_path,
            test_list=[{"test_name": "", "parquet": "target_1000.parquet"}],
            output_category="delta_metrics",
            test_path=test_path,
            metric_name=variant,
            parquet_source_path=Path(gen_path) / "source_1000.parquet",
        )

    generate_job(
        processor_name="domain_gap",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=[
            {
                "test_name": "wasserstein_bytes",
                "parquet": "target_bytes.parquet",
            }
        ],
        output_category="delta_metrics",
        test_path=test_path,
        metric_name="wasserstein_1d",
        parquet_source_path=Path(test_path) / f"{_OUTPUT_DATA_DIR}/source_bytes.parquet",
    )


@pytest.fixture(scope="session")
def job_diversity(test_path: str, diversity_data: Any) -> None:
    """Generate test job configurations for diversity tests.

    Args:
        test_path: Path to the tests directory.
        diversity_data: Fixture that generates diversity test parquet.
    """
    test_list = [
        {"test_name": "diversity", "parquet": _DIVERSITY_PARQUET},
        {"test_name": "diversity_batch", "parquet": _DIVERSITY_PARQUET},
        {"test_name": "diversity_single_metric", "parquet": _DIVERSITY_PARQUET},
    ]

    generate_job(
        processor_name="diversity",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=test_list,
        output_category="metrics",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_features_embeddings(test_path: str, visual_features_data: Any) -> None:
    """Generate test job configurations for features_embeddings tests.

    Args:
        test_path: Path to the tests directory.
        visual_features_data: Fixture that generates visual features parquets.
    """
    test_list = [
        {"test_name": "features_embeddings", "parquet": _VISUAL_FEATURES_PARQUET},
        {
            "test_name": "features_embeddings_batch",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_multi_layer",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_n_layer_0",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_custom_norm",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_prefix",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_suffix",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "features_embeddings_infer_batch_size",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
    ]

    generate_job(
        processor_name="features_embeddings",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=test_list,
        output_category="features",
        test_path=test_path,
    )


@pytest.fixture(scope="session")
def job_visual_features(test_path: str, visual_features_data: Any) -> None:
    """Generate test job configurations for visual features tests.

    Args:
        test_path: Path to the tests directory.
        visual_features_data: Fixture that generates visual features parquets.
    """
    test_list = [
        {"test_name": "visual_features", "parquet": _VISUAL_FEATURES_PARQUET},
        {
            "test_name": "visual_features_batch",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "visual_features_path",
            "parquet": "visual_features_path.parquet",
        },
        {
            "test_name": "visual_features_prefix",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
        {
            "test_name": "visual_features_grayscale_false",
            "parquet": _VISUAL_FEATURES_PARQUET,
        },
    ]

    generate_job(
        processor_name="visual_features",
        parquets_path=Path(test_path) / _OUTPUT_DATA_DIR,
        test_list=test_list,
        output_category="features",
        test_path=test_path,
    )
