"""Pipeline config builder for multi-processor integration tests.

Provides a dict-based config builder that constructs full pipeline YAML
configs with arbitrary processor combinations and root key orderings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProcessorSpec:
    """Specification for a single processor in a pipeline config.

    Attributes:
        name: Processor instance name (e.g., "embeddings", "completeness").
        type: Processor type string (e.g., "features_embeddings", "completeness").
        interface: Which top-level key to place under: "features", "metrics", "gap".
        config: Arbitrary processor-specific configuration dict.
    """

    name: str
    type: str
    interface: str
    config: dict[str, Any] = field(default_factory=dict)


_RESNET18_INFER = {
    "batch_size": 10,
    "width": 64,
    "height": 64,
    "norm_mean": [0.485, 0.456, 0.406],
    "norm_std": [0.229, 0.224, 0.225],
}


def _make_features_embeddings(name: str = "embeddings", input_col: str = "image_bytes") -> ProcessorSpec:
    """Create a ProcessorSpec for features_embeddings (ResNet18 embeddings).

    Args:
        name: Processor instance name.
        input_col: Input column containing image data (bytes or paths).

    Returns:
        ProcessorSpec configured for ResNet18 layer -2 embeddings on CPU.
    """
    return ProcessorSpec(
        name=name,
        type="features_embeddings",
        interface="features",
        config={
            "columns": {"input": [input_col]},
            "model": {"arch": "resnet18", "n_layer_feature": -2, "device": "cpu"},
            "infer": dict(_RESNET18_INFER),
        },
    )


def _make_visual_features(name: str = "visual_features", input_col: str = "image_bytes") -> ProcessorSpec:
    """Create a ProcessorSpec for visual features (luminosity, contrast, blur, entropy).

    Args:
        name: Processor instance name.
        input_col: Input column containing image bytes.

    Returns:
        ProcessorSpec configured for standard visual feature extraction.
    """
    return ProcessorSpec(
        name=name,
        type="image_features",
        interface="features",
        config={
            "columns": {"input": [input_col]},
            "features": ["contrast", "blur", "luminosity", "entropy"],
        },
    )


def _make_completeness(
    name: str = "completeness",
    columns: list[str] | None = None,
) -> ProcessorSpec:
    """Create a ProcessorSpec for completeness metric.

    Args:
        name: Processor instance name.
        columns: Columns to check for completeness (default: visual feature columns).

    Returns:
        ProcessorSpec configured for completeness computation.
    """
    return ProcessorSpec(
        name=name,
        type="completeness",
        interface="metrics",
        config={
            "columns": {"input": columns or ["blur_score", "contrast", "quality_score"]},
        },
    )


def _make_diversity(
    name: str = "diversity",
    columns: list[str] | None = None,
    metrics: list[str] | None = None,
) -> ProcessorSpec:
    """Create a ProcessorSpec for diversity metric.

    Args:
        name: Processor instance name.
        columns: Categorical columns to compute diversity on (default: ["class_name"]).
        metrics: Diversity metrics to compute (default: all available).

    Returns:
        ProcessorSpec configured for diversity computation.
    """
    spec: dict[str, Any] = {"columns": {"input": columns or ["class_name"]}}
    if metrics:
        spec["metrics"] = metrics
    return ProcessorSpec(name=name, type="diversity", interface="metrics", config=spec)


def _make_representativeness(
    name: str = "representativeness",
    columns: list[str] | None = None,
    distribution: str = "uniform",
) -> ProcessorSpec:
    """Create a ProcessorSpec for representativeness metric.

    Args:
        name: Processor instance name.
        columns: Numeric columns to test distribution (default: ["brightness"]).
        distribution: Expected distribution type: "uniform" or "normal".

    Returns:
        ProcessorSpec configured for representativeness testing.
    """
    return ProcessorSpec(
        name=name,
        type="representativeness",
        interface="metrics",
        config={
            "columns": {"input": columns or ["brightness"]},
            "distribution": distribution,
        },
    )


def _make_domain_gap(
    name: str = "domain_gap",
    input_col: str = "image_bytes_embedding",
    metric: str = "mmd_linear",
    **extra: Any,
) -> ProcessorSpec:
    """Create a ProcessorSpec for domain gap metric.

    Args:
        name: Processor instance name.
        input_col: Column containing embeddings to compare.
        metric: Distance metric (mmd_linear, wasserstein_1d, klmvn_diag, etc.).
        **extra: Additional config options passed through to distance config.

    Returns:
        ProcessorSpec configured for domain gap computation.
    """
    config: dict[str, Any] = {
        "columns": {"input": [input_col]},
        "distance": {"metric": metric},
    }
    config.update(extra)
    return ProcessorSpec(name=name, type="domain_gap", interface="gap", config=config)


def _make_domain_gap_split(
    name: str = "domain_gap",
    input_col: str = "image_bytes_embedding",
    metric: str = "mmd_linear",
    split_by: str = "source",
    split_values: list[str] | None = None,
    **extra: Any,
) -> ProcessorSpec:
    """Create a ProcessorSpec for domain gap with split configuration.

    Args:
        name: Processor instance name.
        input_col: Column containing embeddings to compare.
        metric: Distance metric for gap computation.
        split_by: Column to split by for multi-source comparison.
        split_values: Specific split values to include (None = all).
        **extra: Additional config options.

    Returns:
        ProcessorSpec for domain gap with split configuration.
    """
    config: dict[str, Any] = {
        "columns": {"input": [input_col]},
        "distance": {"metric": metric},
    }
    config.update(extra)
    return ProcessorSpec(
        name=name,
        type="domain_gap",
        interface="gap",
        config=config,
    )


def _make_domain_gap_filter(
    name: str = "domain_gap",
    input_col: str = "image_bytes_embedding",
    metric: str = "mmd_linear",
    **extra: Any,
) -> ProcessorSpec:
    """Create a ProcessorSpec for domain gap (alias for _make_domain_gap).

    Args:
        name: Processor instance name.
        input_col: Column containing embeddings to compare.
        metric: Distance metric for gap computation.
        **extra: Additional config options.

    Returns:
        ProcessorSpec for domain gap computation.
    """
    return _make_domain_gap(name=name, input_col=input_col, metric=metric, **extra)


def make_loader(
    data_path: str,
    name: str = "source_dataset",
    batch_size: int = 100,
    sample_path_column: str | None = None,
    split: dict[str, Any] | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a dataloader config dict.

    Args:
        data_path: Path to the input parquet file.
        name: Loader name.
        batch_size: Batch size.
        sample_path_column: If set, adds ``sample_path`` entry.
        split: Split config dict, e.g. ``{"by": "source", "values": ["studio", "outdoor", "zoo"]}``.
        filters: List of filter configs, e.g. ``[{"column": "source", "values": ["outdoor"]}]``.

    Returns:
        Loader config dict.
    """
    loader: dict[str, Any] = {
        "name": name,
        "type": "parquet",
        "path": data_path,
        "batch_size": batch_size,
    }
    if sample_path_column:
        loader["sample_path"] = [{"column": sample_path_column}]
    if split:
        loader["split"] = split
    if filters:
        loader["filters"] = filters
    return loader


def build_pipeline_config(
    data_path: str,
    processors: list[ProcessorSpec],
    root_key_order: list[str],
    output_dir: str,
    config_name: str = "pipeline_test",
    loaders: list[dict[str, Any]] | None = None,
) -> Path:
    """Build a full pipeline config YAML and write it to the generated configs directory.

    Processors are grouped by interface.  Each interface that has processors
    gets its own ``outputs.path`` in ``output_dir``.  The ``root_key_order``
    controls the top-level YAML key ordering (purely cosmetic — topological
    sort determines actual execution order).

    Args:
        data_path: Path to the input parquet file (used when ``loaders`` is
            not specified).
        processors: List of processor specs.
        root_key_order: Order of root keys in the YAML file.
            Valid values: "dataloaders", "features", "metrics", "gap",
            plus optional "storage", "compute", "errors".
        output_dir: Directory for output parquet files.
        config_name: Base name for the generated YAML (without .yaml).
        loaders: Optional list of loader config dicts.  If not provided,
            a single default loader is created from ``data_path``.

    Returns:
        Path to the generated YAML config file.
    """
    groups: dict[str, list[dict[str, Any]]] = {"features": [], "metrics": [], "gap": []}
    for p in processors:
        if p.interface in groups:
            entry: dict[str, Any] = {"name": p.name, "type": p.type}
            entry.update(p.config)
            groups[p.interface].append(entry)

    config: dict[str, Any] = {}

    for key in root_key_order:
        if key == "dataloaders":
            if loaders is not None:
                config["dataloaders"] = {"loaders": loaders}
            else:
                config["dataloaders"] = {"loaders": [make_loader(data_path=data_path)]}

        elif groups.get(key):
            interface_cfg: dict[str, Any] = {
                "processors": groups[key],
            }
            interface_cfg["outputs"] = {"path": str(Path(output_dir) / f"{config_name}_{key}.parquet")}
            config[key] = interface_cfg

        elif key in ("storage", "compute", "errors"):
            config[key] = {}

    config_dir = Path(output_dir).parent.parent / "integration" / "fixtures" / "config" / "generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_name}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ---------------------------------------------------------------------------
# Scenario builders — convenience constructors for common test configs
# ---------------------------------------------------------------------------


def scenario_overview_chain(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Full 8-step chain from examples/overview.md.

    Returns (processors, config_name).
    """
    processors = [
        _make_visual_features(),
        _make_features_embeddings(),
        _make_completeness(),
        _make_diversity(columns=["class_name"]),
        _make_representativeness(columns=["brightness"]),
        _make_domain_gap_split(input_col="image_bytes_embedding", metric="mmd_linear"),
    ]
    return processors, "scenario_overview_chain"


def scenario_quality_gate(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Completeness on critical columns as a quality gate."""
    processors = [
        _make_completeness(columns=["blur_score", "contrast", "quality_score", "brightness"]),
    ]
    return processors, "scenario_quality_gate"


def scenario_class_imbalance(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Diversity by class_name across source cohorts."""
    processors = [
        _make_diversity(columns=["class_name"]),
    ]
    return processors, "scenario_class_imbalance"


def scenario_preprocessing_sanity(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Representativeness on raw tabular columns."""
    processors = [
        _make_representativeness(columns=["brightness", "sharpness"], distribution="uniform"),
    ]
    return processors, "scenario_preprocessing_sanity"


def scenario_train_test_drift(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Embeddings → domain_gap split by sample_type (train/test)."""
    processors = [
        _make_features_embeddings(),
        _make_domain_gap_split(input_col="image_bytes_embedding", metric="mmd_linear"),
    ]
    return processors, "scenario_train_test_drift"


def scenario_acquisition_drift(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Embeddings → domain_gap split by source (environmental drift)."""
    processors = [
        _make_features_embeddings(),
        _make_domain_gap_split(input_col="image_bytes_embedding", metric="mmd_linear"),
    ]
    return processors, "scenario_acquisition_drift"


def scenario_multi_source_vf_diversity(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Visual features → diversity on VF columns, stratified by source."""
    processors = [
        _make_visual_features(input_col="image_bytes"),
        _make_diversity(
            columns=["image_bytes_contrast", "image_bytes_blur"],
        ),
    ]
    return processors, "scenario_multi_source_vf_diversity"


def scenario_feature_selection_assist(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Completeness + representativeness on the same columns."""
    cols = ["brightness", "sharpness"]
    processors = [
        _make_completeness(columns=cols),
        _make_representativeness(columns=cols, distribution="uniform"),
    ]
    return processors, "scenario_feature_selection_assist"


_VF_COLS = [
    "image_bytes_luminosity",
    "image_bytes_contrast",
    "image_bytes_blur",
    "image_bytes_entropy",
]


def scenario_full_story(data_path: str) -> tuple[list[ProcessorSpec], str]:
    """Full end-to-end pipeline matching ``examples/config/full_story.yaml``.

    Includes visual features, ResNet-18 embeddings, completeness, diversity,
    representativeness (all 4 metrics, normal distribution, 20 bins), and
    3 domain gap metrics (FID, MMD-RBF, Wasserstein).

    Returns:
        Tuple of (processors, config_name).
    """
    processors = [
        _make_visual_features(input_col="image_bytes"),
        _make_features_embeddings(name="embedding", input_col="image_bytes"),
        ProcessorSpec(
            name="completeness",
            type="completeness",
            interface="metrics",
            config={
                "columns": {
                    "input": ["quality_score"] + _VF_COLS,
                },
                "include_per_column": True,
                "include_overall": True,
            },
        ),
        _make_diversity(columns=["class_name"]),
        ProcessorSpec(
            name="representativeness",
            type="representativeness",
            interface="metrics",
            config={
                "columns": {"input": _VF_COLS},
                "metrics": ["chi-square", "grte", "shannon-entropy", "kolmogorov-smirnov"],
                "distribution": "normal",
                "histogram": {"bins": 20},
            },
        ),
        _make_domain_gap(
            name="fid_gap",
            input_col="image_bytes_embedding",
            metric="fid",
            distance={"metric": "fid", "epsilon": 1e-6},
        ),
        _make_domain_gap(
            name="mmd_rbf_gap",
            input_col="image_bytes_embedding",
            metric="mmd_rbf",
            distance={"metric": "mmd_rbf", "kernel_params": {"gamma": 1.0}},
        ),
        _make_domain_gap(
            name="wasserstein_gap",
            input_col="image_bytes_embedding",
            metric="wasserstein_1d",
        ),
    ]
    return processors, "scenario_full_story"
