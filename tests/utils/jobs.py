"""Job utility functions for DQM-ML tests.

This module provides helper functions for generating test job configurations.
"""

from pathlib import Path
from typing import Any

import ruamel.yaml

# Domain gap infer parameters per metric: batch_size, width, height
DOMAIN_GAP_INFER_PARAMS = {
    "fid": {"batch_size": 32, "width": 64, "height": 64},
    "klmvn_diag": {"batch_size": 10, "width": 20, "height": 20},
    "mmd_linear": {"batch_size": 10, "width": 64, "height": 64},
    "wasserstein_1d": {"batch_size": 18, "width": 64, "height": 64},
    "mmd_rbf": {"batch_size": 10, "width": 64, "height": 64},
    "mmd_poly": {"batch_size": 10, "width": 64, "height": 64},
    "pad": {"batch_size": 10, "width": 64, "height": 64},
    "cmd": {"batch_size": 50, "width": 64, "height": 64},
}

# Domain gap variant suffixes for testing config parameters.
# Maps suffix → (base_metric, function_to_apply_to_gap_proc)
_DOMAIN_GAP_VARIANTS: dict[str, tuple[str, Any]] = {
    "no_sum_outer": ("fid", lambda p: p.setdefault("summary", {}).__setitem__("collect_sum_outer", False)),
    "no_store": ("mmd_rbf", lambda p: p.setdefault("summary", {}).__setitem__("store_embeddings", False)),
    "mae": ("pad", lambda p: p["distance"].__setitem__("evaluator", "mae")),
    "gamma2": ("mmd_rbf", lambda p: p["distance"].__setitem__("kernel_params", {"gamma": 2.0})),
    "custom_hist": (
        "wasserstein_1d",
        lambda p: p.setdefault("summary", {}).__setitem__("histogram", {"dims": 32, "bins": 16, "range": [-2.0, 2.0]}),
    ),
}


def _parse_domain_gap_test_name(metric_name: str) -> tuple[str, str | None]:
    """Extract base metric and variant suffix from a domain gap test name.

    Args:
        metric_name: Full test name (e.g. ``"fid_no_sum_outer"``, ``"mmd_rbf"``).

    Returns:
        Tuple of (base_metric, variant_suffix_or_None).
    """
    for suffix, (base_metric, _) in _DOMAIN_GAP_VARIANTS.items():
        if metric_name.endswith(f"_{suffix}"):
            return base_metric, suffix
    return metric_name, None


# Command multi-layer features for ResNet
CMD_MULTI_LAYER = [
    "maxpool",
    "layer1.1.relu_1",
    "layer2.1.relu_1",
    "layer3.1.relu_1",
    "layer4.1.relu_1",
]

# Features embeddings variant suffixes for testing config parameters
_FEATURES_EMBEDDINGS_VARIANTS: dict[str, Any] = {
    "multi_layer": lambda p: p.setdefault("model", {}).__setitem__("n_layer_feature", ["maxpool", "layer4.1.relu_1"]),
    "n_layer_0": lambda p: p.setdefault("model", {}).__setitem__("n_layer_feature", -1),
    "custom_norm": lambda p: p.setdefault("infer", {}).update(
        {"norm_mean": [0.0, 0.0, 0.0], "norm_std": [1.0, 1.0, 1.0]}
    ),
    "prefix": lambda p: p.setdefault("columns", {}).__setitem__("prefix", "pfx_"),
    "suffix": lambda p: p.setdefault("columns", {}).__setitem__("suffix", "_sfx"),
    "infer_batch_size": lambda p: p.setdefault("infer", {}).__setitem__("batch_size", 16),
}

# Visual features variant suffixes for testing config parameters
_VISUAL_FEATURES_VARIANTS: dict[str, Any] = {
    "prefix": lambda p: p.setdefault("columns", {}).__setitem__("prefix", "vf_"),
    "grayscale_false": lambda p: p.__setitem__("grayscale", False),
}

# Completeness variant suffixes for testing config parameters
_COMPLETENESS_VARIANTS: dict[str, Any] = {
    "no_per_column": lambda p: p.__setitem__("include_per_column", False),
    "no_overall": lambda p: p.__setitem__("include_overall", False),
}

# Diversity variant suffixes for testing config parameters
_DIVERSITY_VARIANTS: dict[str, Any] = {
    "single_metric": lambda p: p.__setitem__("metrics", ["simpson"]),
}

# Representativeness variant suffixes for testing config parameters
_REPRESENTATIVENESS_VARIANTS: dict[str, Any] = {
    "custom_interpretations": lambda p: p.__setitem__(
        "interpretation",
        {
            "follows_distribution": "matches expected",
            "does_not_follow_distribution": "deviates from expected",
            "high_diversity": "diverse",
            "low_diversity": "uniform values",
            "high_representativeness": "well represented",
            "low_representativeness": "poorly represented",
        },
    ),
    "shannon_threshold": lambda p: p.setdefault("shannon", {}).__setitem__("threshold", 100.0),
}


def _parse_representativeness_test_name(test_name: str) -> tuple[str, str | None]:
    """Parse test name to extract base name and variant suffix.

    Args:
        test_name: Full test name (e.g. ``"uniform_custom_interpretations"``).

    Returns:
        Tuple of (base_test_name, variant_suffix_or_None).
    """
    for suffix in _REPRESENTATIVENESS_VARIANTS:
        if test_name.endswith(f"_{suffix}"):
            return test_name[: -len(f"_{suffix}")], suffix
    return test_name, None


def _parse_features_embeddings_test_name(test_name: str) -> tuple[str, str | None]:
    """Parse test name to extract base name and variant suffix.

    Args:
        test_name: Full test name (e.g. ``"multi_layer"``, ``"n_layer_0"``).

    Returns:
        Tuple of (base_test_name, variant_suffix_or_None).
    """
    for suffix in _FEATURES_EMBEDDINGS_VARIANTS:
        if test_name.endswith(f"_{suffix}"):
            return test_name[: -len(f"_{suffix}")], suffix
    return test_name, None


def _parse_visual_features_test_name(test_name: str) -> tuple[str, str | None]:
    """Parse test name to extract base name and variant suffix.

    Args:
        test_name: Full test name (e.g. ``"path_prefix"``, ``"grayscale_false"``).

    Returns:
        Tuple of (base_test_name, variant_suffix_or_None).
    """
    for suffix in _VISUAL_FEATURES_VARIANTS:
        if test_name.endswith(f"_{suffix}"):
            return test_name[: -len(f"_{suffix}")], suffix
    return test_name, None


def _parse_completeness_test_name(test_name: str) -> tuple[str, str | None]:
    """Parse test name to extract base name and variant suffix.

    Args:
        test_name: Full test name (e.g. ``"no_per_column"``, ``"no_overall"``).

    Returns:
        Tuple of (base_test_name, variant_suffix_or_None).
    """
    for suffix in _COMPLETENESS_VARIANTS:
        if test_name.endswith(f"_{suffix}"):
            return test_name[: -len(f"_{suffix}")], suffix
    return test_name, None


def _parse_diversity_test_name(test_name: str) -> tuple[str, str | None]:
    """Parse test name to extract base name and variant suffix.

    Args:
        test_name: Full test name (e.g. ``"single_metric"``).

    Returns:
        Tuple of (base_test_name, variant_suffix_or_None).
    """
    for suffix in _DIVERSITY_VARIANTS:
        if test_name.endswith(f"_{suffix}"):
            return test_name[: -len(f"_{suffix}")], suffix
    return test_name, None


# Output data directory path
OUTPUT_DATA = "outputs/data"

# Batch sizes per processor for batch tests
BATCH_SIZES = {
    "representativeness": 50000,
    "domain_gap": 50,
    "completeness": 100,
    "diversity": 100,
    "visual_features": 100,
    "features_embeddings": 30,
}

# Map old processor names to their interface keys
PROCESSOR_TO_INTERFACE = {
    "completeness": "metrics",
    "representativeness": "metrics",
    "diversity": "metrics",
    "visual_features": "features",
    "features_embeddings": "features",
    "domain_gap": "gap",
}

# Map old output categories to interface keys
OUTPUT_CATEGORY_TO_INTERFACE = {
    "metrics": "metrics",
    "features": "features",
    "delta_metrics": "gap",
}


def _get_config_name(processor_name: str, test_name: str, metric_name: str | None) -> str:
    """Generate configuration name based on processor and test parameters.

    Args:
        processor_name: Name of the processor (e.g., 'completeness', 'domain_gap').
        test_name: Test configuration name.
        metric_name: Optional metric name for domain gap tests.

    Returns:
        Generated configuration name string.
    """
    if processor_name == "domain_gap":
        return f"{processor_name}_{test_name}" if test_name else f"{processor_name}_{metric_name}"
    if processor_name in ("completeness", "visual_features", "features_embeddings", "diversity"):
        return test_name
    return f"{processor_name}_{test_name}"


def _load_yaml_template(test_path: str, processor_name: str) -> tuple[Any, int, Any]:
    """Load and parse YAML template file.

    Args:
        test_path: Path to the tests directory.
        processor_name: Name of the processor to load template for.

    Returns:
        Tuple of (config_dict, indent, block_seq_indent) from ruamel.yaml.
    """
    template_path = Path(test_path) / f"integration/fixtures/config/templates/{processor_name}.yaml"
    with Path(template_path).open() as file:
        return ruamel.yaml.util.load_yaml_guess_indent(file)  # type: ignore[no-any-return]


def _find_loader(config: dict, loader_name: str) -> dict | None:
    """Find a dataloader by name in the loaders list.

    Args:
        config: Configuration dictionary containing dataloaders section.
        loader_name: Name of the loader to find.

    Returns:
        Loader configuration dict if found, None otherwise.
    """
    for loader in config.get("dataloaders", {}).get("loaders", []):
        if loader.get("name") == loader_name:
            return loader
    return None


def _find_processor(config: dict, interface_key: str, proc_name: str) -> dict | None:
    """Find a processor by name in an interface's processor list.

    Args:
        config: Configuration dictionary containing interfaces.
        interface_key: Interface key (e.g., 'metrics', 'features', 'gap').
        proc_name: Name of the processor to find.

    Returns:
        Processor configuration dict if found, None otherwise.
    """
    interface = config.get(interface_key, {})
    for proc in interface.get("processors", []):
        if proc.get("name") == proc_name:
            return proc
    return None


def _set_loader_paths(config: dict, processor_name: str, parquet_path: Path, parquet_source_path: Path | None) -> None:
    if processor_name == "domain_gap":
        src = _find_loader(config, "source_dataset")
        tgt = _find_loader(config, "target_dataset")
        if src:
            src["path"] = str(parquet_source_path)
        if tgt:
            tgt["path"] = str(parquet_path)
    else:
        loader = _find_loader(config, "source_dataset")
        if loader:
            loader["path"] = str(parquet_path)


def _set_batch_sizes(config: dict, processor_name: str, test_name: str) -> None:
    if "batch" not in test_name:
        return
    batch_size = BATCH_SIZES.get(processor_name)
    if not batch_size:
        return
    loader = _find_loader(config, "source_dataset")
    if loader:
        loader["batch_size"] = batch_size
    if processor_name == "domain_gap":
        tgt = _find_loader(config, "target_dataset")
        if tgt:
            tgt["batch_size"] = batch_size


def _set_sample_path_column(config: dict, test_name: str) -> None:
    src = _find_loader(config, "source_dataset")
    tgt = _find_loader(config, "target_dataset")
    col = "image_bytes" if "bytes" in test_name else "image_path"
    for loader in (src, tgt):
        if loader:
            loader["sample_path"] = [{"column": col}]


def _configure_dataloaders(
    config: dict, processor_name: str, test_name: str, parquet_path: Path, parquet_source_path: Path | None
) -> None:
    _set_loader_paths(config, processor_name, parquet_path, parquet_source_path)
    _set_batch_sizes(config, processor_name, test_name)
    if processor_name == "domain_gap":
        _set_sample_path_column(config, test_name)


def _apply_domain_gap_overrides(gap_proc: dict | None, metric_name: str) -> str | None:
    """Set metric name, apply variant overrides, and configure standard summary blocks.

    Returns:
        The variant suffix, or None if no variant was found.
    """
    base_metric, variant = _parse_domain_gap_test_name(metric_name)
    if gap_proc:
        gap_proc["distance"]["metric"] = base_metric

    if variant and variant in _DOMAIN_GAP_VARIANTS:
        _, apply_fn = _DOMAIN_GAP_VARIANTS[variant]
        apply_fn(gap_proc)

    if gap_proc:
        if metric_name == "fid":
            gap_proc.setdefault("summary", {})["collect_sum_outer"] = True
        elif metric_name == "mmd_rbf":
            gap_proc.setdefault("summary", {})["store_embeddings"] = True

    return base_metric


def _resolve_input_col(config: dict) -> str:
    loaders = config.get("dataloaders", {}).get("loaders", [])
    if loaders:
        modes = loaders[0].get("sample_path", [])
        if modes:
            return modes[0]["column"]
    return "image_path"


def _configure_image_embedding(feat_proc: dict | None, input_col: str, base_metric: str) -> None:
    if not feat_proc:
        return
    feat_proc["columns"] = {"input": [input_col]}
    for param in ("batch_size", "height", "width"):
        if param in DOMAIN_GAP_INFER_PARAMS.get(base_metric, {}):
            feat_proc["infer"][param] = DOMAIN_GAP_INFER_PARAMS[base_metric][param]


def _configure_gap_input_columns(
    gap_proc: dict | None, feat_proc: dict | None, input_col: str, base_metric: str
) -> None:
    if base_metric == "cmd":
        layers = CMD_MULTI_LAYER
        emb_cols = [f"{input_col}_emb_{layer.replace('.', '_')}" for layer in layers]
        if feat_proc:
            feat_proc["model"]["n_layer_feature"] = layers
        if gap_proc:
            gap_proc["columns"]["input"] = emb_cols
            gap_proc["distance"]["feature_weights"] = [1.0] * len(layers)
            gap_proc["distance"]["k"] = 5
    elif gap_proc:
        gap_proc["columns"]["input"] = [f"{input_col}_embedding"]


def _configure_domain_gap(config: dict, processor_name: str, metric_name: str) -> None:
    if processor_name != "domain_gap":
        return

    base_metric = _apply_domain_gap_overrides(_find_processor(config, "gap", "domain_gap"), metric_name)
    input_col = _resolve_input_col(config)
    feat_proc = _find_processor(config, "features", "image_embedding")
    _configure_image_embedding(feat_proc, input_col, base_metric)
    _configure_gap_input_columns(_find_processor(config, "gap", "domain_gap"), feat_proc, input_col, base_metric)


def _apply_variant(variant: str | None, variants_dict: dict[str, Any], proc: dict) -> None:
    if variant and variant in variants_dict:
        variants_dict[variant](proc)


def _configure_representativeness(config: dict, test_name: str) -> None:
    proc = _find_processor(config, "metrics", "representativeness")
    if not proc:
        return
    base_test_name, variant = _parse_representativeness_test_name(test_name)
    _apply_variant(variant, _REPRESENTATIVENESS_VARIANTS, proc)
    proc["distribution"] = "uniform" if "uniform" in base_test_name else "normal"


def _configure_visual_features_proc(config: dict, test_name: str) -> None:
    proc = _find_processor(config, "features", "visual_features")
    if not proc:
        return
    if "path" in test_name:
        proc["columns"]["input"] = ["image_path"]
    _, variant = _parse_visual_features_test_name(test_name)
    _apply_variant(variant, _VISUAL_FEATURES_VARIANTS, proc)


def _configure_features_embeddings_proc(config: dict, test_name: str) -> None:
    proc = _find_processor(config, "features", "features_embeddings")
    if not proc:
        return
    _, variant = _parse_features_embeddings_test_name(test_name)
    _apply_variant(variant, _FEATURES_EMBEDDINGS_VARIANTS, proc)


def _configure_completeness_proc(config: dict, test_name: str) -> None:
    proc = _find_processor(config, "metrics", "completeness")
    if not proc:
        return
    _, variant = _parse_completeness_test_name(test_name)
    _apply_variant(variant, _COMPLETENESS_VARIANTS, proc)


def _configure_diversity_proc(config: dict, test_name: str) -> None:
    proc = _find_processor(config, "metrics", "diversity")
    if not proc:
        return
    _, variant = _parse_diversity_test_name(test_name)
    _apply_variant(variant, _DIVERSITY_VARIANTS, proc)


_PROCESSOR_DISPATCH: dict[str, Any] = {
    "representativeness": _configure_representativeness,
    "visual_features": _configure_visual_features_proc,
    "features_embeddings": _configure_features_embeddings_proc,
    "completeness": _configure_completeness_proc,
    "diversity": _configure_diversity_proc,
}


def _configure_metrics_processor(config: dict, processor_name: str, test_name: str) -> None:
    handler = _PROCESSOR_DISPATCH.get(processor_name)
    if handler:
        handler(config, test_name)


def _configure_output(config: dict, output_category: str, config_name: str, output_path: Path) -> None:
    """Configure output section of the config.

    Args:
        config: Configuration dictionary to modify in place.
        output_category: Output category ('metrics', 'features', 'delta_metrics').
        config_name: Configuration name for output file naming.
        output_path: Base output directory path.
    """
    interface_key = OUTPUT_CATEGORY_TO_INTERFACE.get(output_category)
    if interface_key and interface_key in config:
        interface = config[interface_key]
        if "outputs" not in interface:
            interface["outputs"] = {}
        interface["outputs"]["path"] = f"{output_path!s}/metrics_{config_name}_" + "{}-{}.parquet"


def generate_job(
    test_path: str,
    processor_name: str,
    output_category: str,
    parquets_path: Path,
    test_list: list[dict[str, str]],
    metric_name: str | None = None,
    parquet_source_path: Path | None = None,
) -> None:
    """Generate test job configuration files from templates.

    Args:
        test_path: Path to the tests directory.
        processor_name: Name of the processor (e.g., 'completeness', 'representativeness').
        output_category: Output category (e.g., 'metrics', 'delta_metrics', 'features').
        parquets_path: Path to parquet files directory.
        test_list: List of test configurations to generate.
        metric_name: Optional metric name for domain gap tests.
        parquet_source_path: Optional source parquet path for domain gap tests.
    """
    configs_path = Path(test_path) / "integration/fixtures/config/generated"
    output_path = Path(test_path) / OUTPUT_DATA
    Path(configs_path).mkdir(exist_ok=True, parents=True)

    for test in test_list:
        parquet_path = parquets_path / test["parquet"]
        test_name = test["test_name"]
        config_name = _get_config_name(processor_name, test_name, metric_name)
        config_path = Path(f"{configs_path}/{config_name}.yaml")

        full_config, ind, bsi = _load_yaml_template(test_path, processor_name)

        _configure_dataloaders(full_config, processor_name, test_name, parquet_path, parquet_source_path)
        if metric_name:
            _configure_domain_gap(full_config, processor_name, metric_name)
        _configure_metrics_processor(full_config, processor_name, test_name)
        _configure_output(full_config, output_category, config_name, output_path)

        yaml_config = ruamel.yaml.YAML()
        yaml_config.indent(mapping=ind, sequence=ind, offset=bsi)
        with Path(config_path).open("w") as fp:
            yaml_config.dump(full_config, fp)
