"""Benchmark tests for domain gap metrics using COCO-500 data.

Records timing and computed values for reference, without hard assertions.
"""

from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from dqm_ml_job.cli import execute

_INFER_PARAMS = {
    "fid": {"batch_size": 32, "width": 299, "height": 299},
    "mmd_linear": {"batch_size": 10, "width": 224, "height": 224},
    "mmd_rbf": {"batch_size": 10, "width": 224, "height": 224},
    "mmd_poly": {"batch_size": 10, "width": 224, "height": 224},
    "wasserstein_1d": {"batch_size": 18, "width": 299, "height": 299},
    "klmvn_diag": {"batch_size": 10, "width": 20, "height": 20},
    "pad": {"batch_size": 10, "width": 224, "height": 224},
    "cmd": {"batch_size": 50, "width": 224, "height": 224},
}

_KERNEL_PARAMS = {
    "mmd_rbf": {"gamma": 1.0},
    "mmd_poly": {"degree": 3.0, "gamma": 1.0, "coefficient0": 1.0},
}

_MODEL_ARCH = {
    "fid": "inception_v3",
}


def _build_domain_gap_config(
    output_path: Path,
    config_dir: Path,
    metric: str,
    n_layer_feature: Any = -2,
    embedding_cols: list[str] | None = None,
    feature_weights: list[float] | None = None,
    cmd_k: int | None = None,
) -> Path:
    """Build an inline YAML config for a single domain gap metric.

    Args:
        output_path: Path for output parquets.
        config_dir: Directory to write the generated config.
        metric: Domain-gap metric name.
        n_layer_feature: Layer specification for ImageEmbeddingProcessor.
        embedding_cols: Override input.embedding_cols (for CMD multi-layer).
        feature_weights: Per-layer weights (CMD only).
        cmd_k: Number of moments (CMD only).

    Returns:
        Path to the generated YAML config file.
    """
    infer = _INFER_PARAMS[metric]

    config: dict = {
        "config": {
            "dataloaders": {
                "source_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "source_500.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
                "target_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "target_500.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
            },
            "metrics_processor": {
                "image_embedding": {
                    "type": "image_embedding",
                    "data": {
                        "image_column": "image_path",
                        "mode": "path",
                    },
                    "model": {
                        "arch": _MODEL_ARCH.get(metric, "resnet18"),
                        "n_layer_feature": n_layer_feature,
                        "device": "cpu",
                    },
                    "infer": {
                        "batch_size": infer["batch_size"],
                        "width": infer["width"],
                        "height": infer["height"],
                        "norm_mean": [0.485, 0.456, 0.406],
                        "norm_std": [0.229, 0.224, 0.225],
                    },
                },
                "domain_gap": {
                    "type": "domain_gap",
                    "input": {
                        "embedding_col": "embedding",
                    },
                    "summary": {
                        "store_embeddings": True,
                    },
                    "delta": {
                        "metric": metric,
                    },
                },
            },
            "compute_delta": True,
            "outputs": {
                "delta_metrics": {
                    "type": "parquet",
                    "path_pattern": str(output_path / f"metrics_benchmark_{metric}_") + "{}-{}.parquet",
                    "columns": [],
                },
            },
        }
    }

    # Multi-layer for CMD
    if embedding_cols is not None:
        config["config"]["metrics_processor"]["domain_gap"]["input"]["embedding_cols"] = embedding_cols
        config["config"]["metrics_processor"]["domain_gap"]["input"].pop("embedding_col", None)
    if feature_weights is not None:
        config["config"]["metrics_processor"]["domain_gap"]["delta"]["feature_weights"] = feature_weights
    if cmd_k is not None:
        config["config"]["metrics_processor"]["domain_gap"]["delta"]["k"] = cmd_k

    # Kernel params for MMD-RBF / MMD-Poly
    if metric in _KERNEL_PARAMS:
        config["config"]["metrics_processor"]["domain_gap"]["delta"]["kernel_params"] = _KERNEL_PARAMS[metric]

    # PAD evaluator
    if metric == "pad":
        config["config"]["metrics_processor"]["domain_gap"]["method"] = {"evaluator": "mse"}

    # Write config
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"benchmark_{metric}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "metric",
    [
        "fid",
        "mmd_linear",
        "mmd_rbf",
        "mmd_poly",
        "wasserstein_1d",
        "klmvn_diag",
        "pad",
    ],
)
def test_domain_gap_benchmark(
    test_path: str,
    output_path: Path,
    coco_data_500: None,
    metric: str,
) -> None:
    """Benchmark a single domain gap metric on COCO-500.

    Prints timing and computed value. No hard assertion — the purpose
    is to establish reference values and track performance.

    Args:
        test_path: Path to the tests directory.
        output_path: Path to test output data directory.
        coco_data_500: Fixture providing 500-image parquets.
        metric: Name of the domain gap metric to benchmark.
    """
    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_path = _build_domain_gap_config(output_path, config_dir, metric)

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start

    # Read output
    output_file = f"metrics_benchmark_{metric}_delta-.parquet"
    table = pq.read_table(output_path / output_file)
    df = table.to_pandas()
    value = df[metric].iloc[0]

    print(f"[BENCHMARK] {metric} = {value}  |  time = {elapsed:.2f}s")


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_domain_gap_benchmark_cmd(
    test_path: str,
    output_path: Path,
    coco_data_500: None,
) -> None:
    """Benchmark CMD (Central Moment Discrepancy) on COCO-500.

    CMD requires multi-layer feature extraction. Uses the same five
    ResNet-18 layers as the legacy benchmark:
    maxpool, layer1.1.relu_1, layer2.1.relu_1, layer3.1.relu_1, layer4.1.relu_1

    Args:
        test_path: Path to the tests directory.
        output_path: Path to test output data directory.
        coco_data_500: Fixture providing 500-image parquets.
    """
    layers = [
        "maxpool",
        "layer1.1.relu_1",
        "layer2.1.relu_1",
        "layer3.1.relu_1",
        "layer4.1.relu_1",
    ]
    emb_cols = [f"emb_{layer.replace('.', '_')}" for layer in layers]

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_path = _build_domain_gap_config(
        output_path,
        config_dir,
        metric="cmd",
        n_layer_feature=layers,
        embedding_cols=emb_cols,
        feature_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        cmd_k=5,
    )

    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start

    # Read output
    output_file = "metrics_benchmark_cmd_delta-.parquet"
    table = pq.read_table(output_path / output_file)
    df = table.to_pandas()
    value = df["cmd"].iloc[0]

    print(f"[BENCHMARK] cmd = {value}  |  time = {elapsed:.2f}s")


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_domain_gap_benchmark_all_metrics(
    test_path: str,
    output_path: Path,
    coco_data_500: None,
) -> None:
    """Benchmark all domain gap metrics in a single job on COCO-500.

    This test runs all eight domain gap metrics (FID, MMD-linear, MMD-RBF, MMD-Poly,
    Wasserstein-1D, KLMVN-diag, PAD, CMD) in a single job execution, using a shared
    image_embedding processor (ResNet-18, 224x224) for efficiency.

    Note: This uses ResNet-18 for all metrics, including FID and Wasserstein-1D, which
    in their individual benchmarks use different models and image sizes. The values
    are for reference only and not expected to match the individual benchmarks.

    Args:
        test_path: Path to the tests directory.
        output_path: Path to test output data directory.
        coco_data_500: Fixture providing 500-image parquets.
    """
    # We'll use a fixed image size and batch size for all metrics
    common_width = 224
    common_height = 224
    common_batch_size = 32

    config_dir = Path(test_path) / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "config": {
            "dataloaders": {
                "source_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "source_500.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
                "target_dataset": {
                    "type": "parquet",
                    "path": str(output_path / "target_500.parquet"),
                    "batch_size": 50,
                    "memory_limit": "2GB",
                    "threads": 4,
                },
            },
            "metrics_processor": {
                "image_embedding": {
                    "type": "image_embedding",
                    "data": {
                        "image_column": "image_path",
                        "mode": "path",
                    },
                    "model": {
                        "arch": "resnet18",
                        "n_layer_feature": -2,
                        "device": "cpu",
                    },
                    "infer": {
                        "batch_size": common_batch_size,
                        "width": common_width,
                        "height": common_height,
                        "norm_mean": [0.485, 0.456, 0.406],
                        "norm_std": [0.229, 0.224, 0.225],
                    },
                },
                # Second ImageEmbeddingProcessor for CMD multi-layer features.
                # CMD needs 5 ResNet-18 layer columns; the shared processor only
                # produces a single avgpool embedding for other metrics.
                "image_embedding_cmd": {
                    "type": "image_embedding",
                    "data": {
                        "image_column": "image_path",
                        "mode": "path",
                    },
                    "model": {
                        "arch": "resnet18",
                        "n_layer_feature": [
                            "maxpool",
                            "layer1.1.relu_1",
                            "layer2.1.relu_1",
                            "layer3.1.relu_1",
                            "layer4.1.relu_1",
                        ],
                        "device": "cpu",
                    },
                    "infer": {
                        "batch_size": common_batch_size,
                        "width": common_width,
                        "height": common_height,
                        "norm_mean": [0.485, 0.456, 0.406],
                        "norm_std": [0.229, 0.224, 0.225],
                    },
                },
            },
            "compute_delta": True,
            "outputs": {
                "delta_metrics": {
                    "type": "parquet",
                    "path_pattern": str(output_path / "metrics_benchmark_all.parquet"),
                    "columns": [],
                },
            },
        }
    }

    # We'll add a domain_gap processor for each metric
    metrics = [
        "fid",
        "mmd_linear",
        "mmd_rbf",
        "mmd_poly",
        "wasserstein_1d",
        "klmvn_diag",
        "pad",
        "cmd",
    ]

    for _i, metric in enumerate(metrics):
        # Use a unique name for each domain_gap processor
        processor_name = f"domain_gap_{metric}"
        config["config"]["metrics_processor"][processor_name] = {
            "type": "domain_gap",
            "input": {
                "embedding_col": "embedding",
            },
            "summary": {
                # We'll set store_embeddings based on the metric
                "store_embeddings": metric in {"mmd_rbf", "mmd_poly", "pad", "cmd"},
            },
            "delta": {
                "metric": metric,
            },
        }
        # Add kernel parameters if needed
        if metric in _KERNEL_PARAMS:
            config["config"]["metrics_processor"][processor_name]["delta"]["kernel_params"] = _KERNEL_PARAMS[metric]
        # Add CMD-specific parameters
        if metric == "cmd":
            # We need to specify the embedding_cols for CMD (multi-layer)
            # We'll use the same five layers as in the individual CMD benchmark
            layers = [
                "maxpool",
                "layer1.1.relu_1",
                "layer2.1.relu_1",
                "layer3.1.relu_1",
                "layer4.1.relu_1",
            ]
            emb_cols = [f"emb_{layer.replace('.', '_')}" for layer in layers]
            config["config"]["metrics_processor"][processor_name]["input"]["embedding_cols"] = emb_cols
            config["config"]["metrics_processor"][processor_name]["input"].pop("embedding_col", None)
            config["config"]["metrics_processor"][processor_name]["delta"]["feature_weights"] = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            config["config"]["metrics_processor"][processor_name]["delta"]["k"] = 5
        # Add PAD evaluator
        if metric == "pad":
            config["config"]["metrics_processor"][processor_name]["method"] = {"evaluator": "mse"}

    # Write the config
    config_path = config_dir / "benchmark_all_metrics.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Execute the job
    start = timer()
    execute(["-p", str(config_path)])
    elapsed = timer() - start

    # Read the output
    output_file = "metrics_benchmark_all.parquet"
    table = pq.read_table(output_path / output_file)
    df = table.to_pandas()

    # Print each metric's value and the time
    for metric in metrics:
        if metric in df.columns:
            # Extract first non-null value (each metric appears in exactly one row)
            value = df[metric].dropna().iloc[0]
            print(f"[BENCHMARK-ALL] {metric} = {value} | time = {elapsed:.2f}s")
        else:
            print(f"[BENCHMARK-ALL] {metric} = MISSING | time = {elapsed:.2f}s")
