import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dqm_ml_core import PluginLoadedRegistry


def _load_metric_from_yaml(config_path: Path) -> tuple[str, dict[str, Any]]:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    metrics_cfg = cfg.get("pipeline_config", {}).get("metrics_processor", {})
    if not metrics_cfg:
        raise ValueError("no metrics_processor found under pipeline_config in config file")
    if len(metrics_cfg) != 1:
        raise ValueError("config must define exactly one metric under metrics_processor for this script")
    metric_key = next(iter(metrics_cfg.keys()))
    metric_cfg = metrics_cfg[metric_key]
    return metric_key, metric_cfg


def _resolve_requested_columns(metric_cfg: dict[str, Any]) -> list[str]:
    requested = list(dict.fromkeys(list(metric_cfg.get("input_columns", [])) + list(metric_cfg.get("columns", []))))
    return requested


def _iter_groups(
    con: duckdb.DuckDBPyConnection, source_path: Path, group_by_cols: list[str]
) -> Iterable[tuple[Any, ...]]:
    cols_expr = ", ".join([f"{c}" for c in group_by_cols])
    query = f"""
        SELECT DISTINCT {cols_expr}
        FROM read_parquet($1)
        WHERE {" AND ".join([f"{c} IS NOT NULL" for c in group_by_cols])}
        ORDER BY {cols_expr}
    """
    for row in con.execute(query, [str(source_path)]).fetchall():
        yield tuple(row)


def _arrow_subset(
    con: duckdb.DuckDBPyConnection,
    source_path: Path,
    needed_cols: list[str],
    group_by_cols: list[str],
    group_values: tuple[Any, ...],
) -> pa.Table:
    where_clauses = [f"{col} = ${i + 2}" for i, col in enumerate(group_by_cols)]
    select_cols = ", ".join([f"{c}" for c in needed_cols])
    query = f"""
        SELECT {select_cols}
        FROM read_parquet($1)
        WHERE {" AND ".join(where_clauses)}
    """
    params: list[Any] = [str(source_path), *group_values]
    return con.execute(query, params).fetch_arrow_table()


def _compute_metric_for_table(
    table: pa.Table, metric_key: str, metric_cfg: dict[str, Any], resolved_cols: list[str]
) -> dict[str, Any]:
    metric_type = metric_cfg.get("type")
    if not metric_type:
        raise ValueError(f"metric '{metric_key}' missing required 'type' in config")
    registry = PluginLoadedRegistry.get_metrics_registry()
    if metric_type not in registry:
        raise ValueError(f"unknown metric type '{metric_type}'. Available: {list(registry.keys())}")
    processor = registry[metric_type](name=metric_key, config=metric_cfg)

    features: dict[str, pa.Array] = {}
    for col in resolved_cols:
        if col in table.schema.names:
            features[col] = table.column(col)

    batch_metrics = processor.compute_batch_metric(features)
    dataset_metrics = processor.compute(batch_metrics)
    return dataset_metrics


def run(
    input_parquet: Path,
    config_path: Path,
    group_by: list[str],
    output_metrics: Path,
    join_output: Path | None = None,
) -> None:
    metric_key, metric_cfg = _load_metric_from_yaml(config_path)
    group_by_cols = list(group_by)
    requested_cols = _resolve_requested_columns(metric_cfg)
    con = duckdb.connect()

    # Discover available columns in the input parquet
    schema_table = con.execute("SELECT * FROM read_parquet($1) LIMIT 0", [str(input_parquet)]).fetch_arrow_table()
    print(schema_table.schema.names)
    available_cols = set(schema_table.schema.names)

    missing_group = [c for c in group_by_cols if c not in available_cols]
    if missing_group:
        raise ValueError(f"Missing group-by columns in file: {missing_group}. Available: {sorted(available_cols)}")

    resolved_cols: list[str] = []
    for col in requested_cols:
        if col in available_cols:
            resolved_cols.append(col)
        elif f"m_{col}" in available_cols:
            resolved_cols.append(f"m_{col}")
        else:
            pass

    if not resolved_cols:
        raise ValueError(
            "None of the requested representativeness columns were found in the file. "
            f"Requested={requested_cols}. Available={sorted(available_cols)}"
        )

    needed_cols = resolved_cols
    metric_cfg_for_subset = dict(metric_cfg)
    if requested_cols:
        metric_cfg_for_subset["input_columns"] = resolved_cols
        metric_cfg_for_subset["columns"] = resolved_cols

    rows: list[dict[str, Any]] = []
    for group_vals in _iter_groups(con, input_parquet, group_by_cols):
        subset = _arrow_subset(con, input_parquet, needed_cols, group_by_cols, group_vals)
        if subset.num_rows == 0:
            continue
        metrics = _compute_metric_for_table(subset, metric_key, metric_cfg_for_subset, resolved_cols)

        # attach group columns
        row: dict[str, Any] = {}
        for i, col in enumerate(group_by_cols):
            row[col] = group_vals[i]
        # flatten metrics dict into row
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    if not rows:
        empty_cols: dict[str, pa.Array] = {c: pa.array([], type=pa.string()) for c in group_by_cols}
        pq.write_table(pa.table(empty_cols), output_metrics)
        return

    out_table = pa.Table.from_pylist(rows)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_metrics)

    if join_output is not None:
        cols_expr = ", ".join([f"{c}" for c in group_by_cols])
        data_path_sql = "'" + str(input_parquet).replace("'", "''") + "'"
        metrics_path_sql = "'" + str(output_metrics).replace("'", "''") + "'"
        join_out_sql = "'" + str(join_output).replace("'", "''") + "'"

        join_sql = (
            f"SELECT d.*, m.* EXCLUDE ({cols_expr}) "
            f"FROM read_parquet({data_path_sql}) d "
            f"LEFT JOIN read_parquet({metrics_path_sql}) m USING ({cols_expr})"
        )
        join_output.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({join_sql}) TO {join_out_sql} (FORMAT PARQUET)")


# %%
def parse_args() -> Any:
    parser = argparse.ArgumentParser(
        prog="duckdb-subsets",
        description="Compute representativeness metrics per group from a single parquet using DuckDB",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input parquet path")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        type=Path,
        default=Path("packages/dqm-ml-pipeline/config/representativness.yaml"),
        help="Pipeline YAML containing representativeness config",
    )
    parser.add_argument(
        "-g",
        "--group-by",
        required=False,
        type=str,
        default="split,class_name",
        help="Comma-separated grouping columns (e.g., split,class_name)",
    )
    parser.add_argument(
        "-o",
        "--output-metrics",
        required=False,
        type=Path,
        default=Path("output/all_metrics.parquet"),
        help="Output parquet path for metrics",
    )
    parser.add_argument(
        "-j",
        "--join-output",
        required=False,
        type=Path,
        help="If set, write a joined dataset parquet with metrics added",
    )
    return parser.parse_args()


def parse_arg_old() -> Any:
    parser = argparse.ArgumentParser(
        prog="dqm-ml-split",
        description="Materialize per-group parquet subsets and delegate metrics computation to dqm-ml pipeline",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input parquet path")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        type=Path,
        default=Path("packages/dqm-ml-pipeline/config/representativness.yaml"),
        help="Pipeline YAML containing representativeness config",
    )
    parser.add_argument(
        "-g",
        "--group-by",
        required=False,
        type=str,
        default="split,class_name",
        help="Comma-separated grouping columns (e.g., split,class_name)",
    )
    parser.add_argument(
        "-o",
        "--output-metrics",
        required=False,
        type=Path,
        default=Path("output/all_metrics.parquet"),
        help="Output parquet path for metrics",
    )
    parser.add_argument(
        "-j",
        "--join-output",
        required=False,
        type=Path,
        help="If set, write a joined dataset parquet with metrics added",
    )
    return parser.parse_args()


def execute() -> None:
    args = parse_args()
    print(args)
    group_by = [c.strip() for c in args.group_by.split(",") if c.strip()]
    print(group_by)

    run(
        input_parquet=args.input,
        config_path=args.config,
        group_by=group_by,
        output_metrics=args.output_metrics,
        join_output=args.join_output,
    )


if __name__ == "__main__":
    execute()
