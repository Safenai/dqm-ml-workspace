"""Dataset job orchestrator for end-to-end data quality assessment.

This module contains the DatasetJob class that orchestrates the complete
pipeline: data loading, metric computation, and result persistence.
"""

import fnmatch
import itertools
import logging
from typing import Any

from dqm_ml_core.api.data_processor import DatametricProcessor
from dqm_ml_core.utils.matching import has_pattern, resolve_include_exclude
import numpy as np
import pyarrow as pa
from tqdm import tqdm

from dqm_ml_job.dataloaders import DataLoader, DataSelection
from dqm_ml_job.outputwriter import OutputWriter

logger = logging.getLogger(__name__)


class DatasetJob:
    """
    Orchestrates the end-to-end data quality assessment process.

    The job handles:
    1. Plugin discovery and component initialization.
    2. Data selection discovery via DataLoaders.
    3. Streaming execution: Iterating over selections and batches to
       compute features and metrics.
    4. Result persistence via OutputWriters.
    5. Comparison metrics (deltas) between discovered datasets.
    """

    def __init__(
        self,
        dataloaders: dict[str, DataLoader],
        metrics: dict[str, DatametricProcessor],
        features_output: OutputWriter | None,
        progress_bar: bool = True,
        threads: int = 4,
        errors_by_interface: dict[str, Any] | None = None,
        compute_seed: int | None = None,
        compute_device: str = "auto",
        compute_max_memory: str | None = None,
    ) -> None:
        """
        Initialize the pipeline components.

        Args:
            dataloaders: Map of initialized DataLoader instances.
            metrics: Map of initialized DatametricProcessor instances.
            features_output: Optional writer for persisting per-sample features.
            progress_bar: Whether to display execution progress in the terminal.
            threads: Number of threads for parallel processing.
            errors_by_interface: Per-interface error configuration.
            compute_seed: Seed for reproducible RNG in processors.
            compute_device: Device hint ("auto", "cpu", "cuda") for processors.
            compute_max_memory: Optional max memory string (e.g. "2GB") for features flushing.
        """
        # We initialize loaded pluging elements
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.features_output = features_output
        self.progress_bar = progress_bar
        self.threads = threads
        self.errors_by_interface = errors_by_interface or {}
        self.compute_max_memory = compute_max_memory

        self._resolve_output_columns()
        self._analyze_processor_columns()

        # Inject per-interface errors into processors
        self._inject_per_interface_errors()

        # Inject compute config into processors
        self._inject_per_interface_compute(compute_seed, compute_device)

        logger.info(
            f"DQM job pipeline initialized will process "
            f"{len(self.dataloaders)} dataloaders, "
            f"{len(self.metrics)} metrics processors, "
            f"{1 if self.features_output else 0} output writers"
        )

    def _resolve_output_columns(self) -> None:
        """Resolve features_output include/exclude columns from the output writer config."""
        self.features_output_include = None
        self.features_output_exclude = None
        if not self.features_output:
            return
        self.features_output_include = self.features_output.columns or None
        self.features_output_exclude = getattr(self.features_output, "exclude", None)

    def _analyze_processor_columns(self) -> None:
        """Collect needed input columns, generated features, and generated metrics from all processors."""
        self.needed_input_columns = []
        self.generated_features = []
        self.generated_metrics = []
        self._has_wildcard_columns = False
        for metric in self.metrics.values():
            cols = metric.needed_columns()
            self.needed_input_columns.extend(cols)
            if not self._has_wildcard_columns:
                self._has_wildcard_columns = any(has_pattern(c) for c in cols)
            self.generated_features.extend(metric.generated_features())
            self.generated_metrics.extend(metric.generated_metrics())

        self.needed_input_columns = list(dict.fromkeys(self.needed_input_columns))
        self.generated_features = list(dict.fromkeys(self.generated_features))
        self.generated_metrics = list(dict.fromkeys(self.generated_metrics))

        if self._has_wildcard_columns:
            self.needed_input_columns = []

        if not self.features_output_include:
            return
        for col in self.features_output_include:
            if has_pattern(col):
                continue
            if col not in self.generated_features:
                logger.info(f"Adding required output column '{col}' to input columns")
                self.needed_input_columns.insert(0, col)

    def _get_interface_for_processor(self, processor_name: str) -> str | None:
        """Determine which interface a processor belongs to.

        Args:
            processor_name: Name of the processor.

        Returns:
            Interface name ("features", "metrics", "gap") or None if unknown.
        """
        # Map processor names to interfaces based on naming patterns
        # This is a heuristic - in a real implementation, you might want
        # to store this information in the processor config itself
        if processor_name in ["visual_features", "image_embedding"]:
            return "features"
        elif processor_name in ["completeness", "diversity", "representativeness"]:
            return "metrics"
        elif processor_name in ["domain_gap"]:
            return "gap"
        return None

    def _inject_per_interface_errors(self) -> None:
        """Inject per-interface errors into processors based on their interface."""
        for metric in self.metrics.values():
            interface = self._get_interface_for_processor(metric.name)
            if interface and interface in self.errors_by_interface:
                metric.errors_config = self.errors_by_interface[interface]

    def _inject_per_interface_compute(self, compute_seed: int | None, compute_device: str) -> None:
        """Inject compute config into processors for device and seed."""
        for metric in self.metrics.values():
            metric.compute_device = compute_device
            if compute_seed is not None:
                metric.compute_seed = compute_seed

    def get_ordered_metrics(self) -> list[DatametricProcessor]:
        """
        Return the list of metrics processors in dependency order.

        Processors that generate columns (via ``generated_features()`` or
        ``generated_columns()``) are placed before processors that depend on
        those columns (via ``needed_columns()``).  This ensures, for example,
        that an ``image_embedding`` processor that produces the ``embedding``
        column runs before a ``domain_gap`` processor that consumes it,
        regardless of the order in which they appear in the YAML config.
        """
        procs = list(self.metrics.values())
        if len(procs) <= 1:
            return procs

        dep_on = self._build_dependency_graph(procs)
        return self._topological_sort(procs, dep_on)

    @staticmethod
    def _register_generated_columns(procs: list[DatametricProcessor]) -> dict[str, set[int]]:
        """Build a mapping from column names to the processor indices that generate them.

        Args:
            procs: List of metric processors.

        Returns:
            Dict mapping column names to sets of processor indices.
        """
        generated_by: dict[str, set[int]] = {}
        for i, p in enumerate(procs):
            for col in p.generated_features():
                generated_by.setdefault(col, set()).add(i)
            if hasattr(p, "generated_columns"):
                for col in p.generated_columns():
                    generated_by.setdefault(col, set()).add(i)
        return generated_by

    @staticmethod
    def _resolve_dependency_col(
        col: str,
        generated_names: list[str],
        generated_by: dict[str, set[int]],
        exclude_idx: int,
    ) -> set[int]:
        """Resolve processor dependencies for a required column.

        Matches the column pattern against generated column names and returns
        indices of processors that produce matching columns (excluding self).

        Args:
            col: Required column name (may contain fnmatch patterns).
            generated_names: List of all column names generated by any processor.
            generated_by: Mapping from column name to set of processor indices.
            exclude_idx: Index of the processor requesting the dependency (excluded).

        Returns:
            Set of processor indices that generate matching columns.
        """
        matching_cols = fnmatch.filter(generated_names, col) if has_pattern(col) else [col]
        deps: set[int] = set()
        for gen_col in matching_cols:
            for gen_idx in generated_by.get(gen_col, ()):
                if gen_idx != exclude_idx:
                    deps.add(gen_idx)
        return deps

    @staticmethod
    def _build_dependency_graph(procs: list[DatametricProcessor]) -> list[set[int]]:
        """Build a dependency graph from a list of processors.

        Args:
            procs: List of metric processors.

        Returns:
            List of sets where dep_on[i] contains indices of processors
            that processor i depends on.
        """
        generated_by = DatasetJob._register_generated_columns(procs)
        generated_names = list(generated_by.keys())
        dep_on: list[set[int]] = [set() for _ in procs]
        for i, p in enumerate(procs):
            for col in p.needed_columns():
                dep_on[i] |= DatasetJob._resolve_dependency_col(col, generated_names, generated_by, i)
        return dep_on

    @staticmethod
    def _topological_sort(procs: list[DatametricProcessor], dep_on: list[set[int]]) -> list[DatametricProcessor]:
        """Topological sort of processors using Kahn's algorithm.

        Args:
            procs: List of metric processors.
            dep_on: Dependency graph as produced by _build_dependency_graph.

        Returns:
            Processors in dependency order.
        """
        ordered: list[DatametricProcessor] = []
        remaining = set(range(len(procs)))
        while remaining:
            ready = {i for i in remaining if not (dep_on[i] & remaining)}
            if not ready:
                ready = {min(remaining)}
            for i in sorted(ready):
                ordered.append(procs[i])
                remaining.remove(i)
        return ordered

    def describe(self, selections: list[DataSelection]) -> None:
        """Log a summary of the execution plan, including selections and metrics."""
        logger.info(f"Executing dqm-ml-job on {len(selections)} selections, using {len(self.metrics)} metrics ")
        for selection in selections:
            logger.info(f"  Selection: {selection.name} -> {selection}")

        for metric_name, metric in self.metrics.items():
            logger.info(f"  Metric: {metric_name} -> {metric}")
            logger.info(f"    Needed columns: {metric.needed_columns()}")
            logger.info(f"    Generated features: {metric.generated_features()}")
            logger.info(f"    Generated metrics: {metric.generated_metrics()}")

    def _discover_selections(self) -> list[DataSelection]:
        """Discover all data selections from all configured dataloaders.

        Returns:
            List of DataSelection instances.
        """
        all_selections: list[DataSelection] = []
        for loader in self.dataloaders.values():
            all_selections.extend(loader.get_selections())
        return all_selections

    def _compute_selection_metrics(
        self,
        selection_name: str,
        batches_metrics_array: dict[str, Any],
        metrics_processors: list[DatametricProcessor],
    ) -> dict[str, Any]:
        """Compute dataset-level metrics for a single selection.

        Args:
            selection_name: Name of the selection.
            batches_metrics_array: Accumulated batch metrics.
            metrics_processors: List of processors.

        Returns:
            Dictionary of computed dataset metrics.
        """
        dataset_metrics: dict[str, Any] = {}
        metrics_iter = (
            tqdm(metrics_processors, desc="metrics", position=1, leave=False)
            if self.progress_bar
            else metrics_processors
        )
        for metric in metrics_iter:
            if logging.getLogger().level == logging.DEBUG:
                logger.debug(f"Metric computation {metric.__class__.__name__} for dataselection {selection_name}")
            dataset_metrics.update(metric.compute(batch_metrics=batches_metrics_array))
            if logging.getLogger().level == logging.DEBUG:
                logger.debug(f"Available metrics  {list(dataset_metrics.keys())}")
        return dataset_metrics

    def run(self) -> tuple[dict[Any, dict[str, Any]], pa.Table | None]:
        """
        Execute the job on all discovered data selections.

        This is the main entry point for execution. It iterates through every
        selection found by the loaders, computes statistics, and finally
        calculates deltas between datasets.

        Returns:
            A tuple containing:
                - Mapping of selection names to their final metric dictionaries.
                - pyarrow Table containing all computed deltas.
        """
        metrics_processors = self.get_ordered_metrics()
        all_selections = self._discover_selections()

        self.describe(all_selections)

        dataselection_metrics_list: dict[Any, dict[str, Any]] = {}
        job_iter = tqdm(all_selections, desc="selection", position=0) if self.progress_bar else all_selections

        for selection in job_iter:
            selection_name = selection.name
            logger.info(f"Processing selection '{selection_name}'")

            selection.bootstrap(self.needed_input_columns)
            batches_metrics_array = self._compute_batches_metrics(selection_name, selection, metrics_processors)

            dataset_metrics = self._compute_selection_metrics(selection_name, batches_metrics_array, metrics_processors)
            dataselection_metrics_list[selection_name] = dataset_metrics

            # Reset processor state between selections — processors like
            # RepresentativenessProcessor cache per-selection state (e.g.
            # quantile bin edges).  Without a reset those cached values
            # leak across selections and produce NaN/incorrect results
            # when the next selection's distribution differs from the first
            # one that was processed.  See AGENTS.md for background.
            for metric in metrics_processors:
                metric.reset()

        delta_metrics_table = self._compute_delta_metrics(metrics_processors, dataselection_metrics_list)

        if self.features_output and hasattr(self.features_output, "flush"):
            self.features_output.flush()

        return dataselection_metrics_list, delta_metrics_table

    @staticmethod
    def _to_pa_array(value: Any, key: str) -> pa.Array:
        """Convert a delta metric value to PyArrow array.

        Args:
            value: The value to convert (float, int, str, np.ndarray, or pa.Array).
            key: The metric name for error logging.

        Returns:
            PyArrow array containing the value.

        Raises:
            TypeError: If the value type is not supported.
        """
        if isinstance(value, pa.Array):
            return value
        elif isinstance(value, (int, float, np.number)):
            return pa.array([float(value)])
        elif isinstance(value, str):
            return pa.array([value])
        elif isinstance(value, np.ndarray):
            return pa.array([value.tolist()])
        else:
            logger.error(f"Cannot convert delta metric '{key}' to pa.Array: type={type(value)}")
            raise TypeError(f"Unsupported delta metric type: {type(value)} for key '{key}'")

    def _compute_delta_metrics(
        self, metrics_processors: list[DatametricProcessor], dataselection_metrics_list: dict[str, dict[str, Any]]
    ) -> pa.Table | None:
        """Compute comparison metrics between every unique pair of data selections.

        Builds a single table with one row per (pair, metric) combination.
        Different metric processors may produce different columns; missing
        values are padded with nulls via ``pa.concat_tables``.

        Args:
            metrics_processors: List of processors capable of computing deltas.
            dataselection_metrics_list: Map of selection names to their metrics.

        Returns:
            A pyarrow Table with one row per (pair, metric) combination.
        """

        selection_combinations = itertools.combinations(dataselection_metrics_list, 2)

        tables: list[pa.Table] = []
        for combination in selection_combinations:
            src_metrics = dataselection_metrics_list[combination[0]]
            target_metrics = dataselection_metrics_list[combination[1]]

            for metric in metrics_processors:
                delta_metrics = metric.compute_delta(src_metrics, target_metrics)

                if len(delta_metrics) == 0:
                    continue

                row = {key: self._to_pa_array(value, key) for key, value in delta_metrics.items()}
                row["selection_source"] = pa.array([combination[0]])
                row["selection_target"] = pa.array([combination[1]])
                tables.append(pa.table(row))

        if not tables:
            return None

        return pa.concat_tables(tables, promote_options="default")

    @staticmethod
    def _process_batch(
        batch: Any, metrics_processors: list[DatametricProcessor]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute features and batch-level metrics for a single batch.

        Args:
            batch: Input data batch.
            metrics_processors: List of processors to apply.

        Returns:
            Tuple of (batch_features, batch_metrics).
        """
        batch_features: dict[str, Any] = {}
        batch_metrics: dict[str, Any] = {}
        for metric in metrics_processors:
            batch_features.update(metric.compute_features(batch, prev_features=batch_features))
            batch_metrics.update(metric.compute_batch_metric(batch_features))
            if logging.getLogger().level == logging.DEBUG:
                m_keys, m_features = list(batch_metrics.keys()), list(batch_features.keys())
                logger.debug(f"{metric.name} - Available batch_metrics  {m_keys} - features {m_features}")
        return batch_features, batch_metrics

    def _accumulate_source_features(
        self,
        batch: Any,
        features_accumulator: dict[str, list[Any]],
        feature_array_size: int,
    ) -> int:
        """Accumulate source dataset columns into the features accumulator.

        Args:
            batch: Input data batch.
            features_accumulator: Dict accumulating feature lists.
            feature_array_size: Current memory usage estimate.

        Returns:
            Updated feature_array_size.
        """
        if self.features_output is None:
            return feature_array_size

        available = batch.column_names
        keep = resolve_include_exclude(
            self.features_output_include,
            self.features_output_exclude,
            available,
        )

        for col_name in keep:
            col_data = batch.column(col_name)
            if col_name not in features_accumulator:
                features_accumulator[col_name] = []
            features_accumulator[col_name].append(col_data)
            feature_array_size += col_data.get_total_buffer_size()
        return feature_array_size

    def _accumulate_generated_features(
        self,
        batch: Any,
        batch_features: dict[str, Any],
        batch_metrics: dict[str, Any],
        features_accumulator: dict[str, list[Any]],
        feature_array_size: int,
    ) -> int:
        """Accumulate generated features into the features accumulator.

        Args:
            batch: Input data batch (used to identify source columns).
            batch_features: Features generated by processors.
            batch_metrics: Metrics generated by processors.
            features_accumulator: Dict accumulating feature lists.
            feature_array_size: Current memory usage estimate.

        Returns:
            Updated feature_array_size.
        """
        if self.features_output is None:
            return feature_array_size

        # Generated features are always included in the output.
        # The include/exclude filter applies only to source columns
        # (handled in _accumulate_source_features).
        source_cols = set(batch.schema.names)

        for k, v in batch_features.items():
            if k in batch_metrics or k in source_cols:
                continue
            if k not in features_accumulator:
                features_accumulator[k] = []
            features_accumulator[k].append(v)
            feature_array_size += v.get_total_buffer_size()
        return feature_array_size

    def _maybe_flush_features(
        self,
        selection_name: str,
        features_accumulator: dict[str, list[Any]],
        feature_array_size: int,
        part_index: int,
        memory_threshold: int,
    ) -> int:
        """Flush features to disk if memory threshold is exceeded.

        Args:
            selection_name: Name of the current data selection.
            features_accumulator: Dict accumulating feature lists (mutated in place on flush).
            feature_array_size: Current memory usage estimate.
            part_index: Current chunk index.
            memory_threshold: Memory threshold in bytes.

        Returns:
            Updated part_index (incremented if flush occurred).
        """
        if feature_array_size <= memory_threshold or not self.features_output:
            return part_index

        logger.info(f"Memory threshold reached ({feature_array_size / 1024**2:.1f}MB). Flushing chunk {part_index}")
        features_chunk: dict[str, Any] = {}
        for k, v_list in features_accumulator.items():
            features_chunk[k] = pa.concat_arrays(v_list)

        self._inject_dataloader_column(selection_name, features_chunk)
        self.features_output.write_table(selection_name, features_chunk, part_index)
        features_accumulator.clear()
        return part_index + 1

    def _write_remaining_features(
        self,
        selection_name: str,
        features_accumulator: dict[str, list[Any]],
        part_index: int,
    ) -> None:
        """Concatenate and write remaining features that were never flushed.

        Args:
            selection_name: Name of the current data selection.
            features_accumulator: Dict accumulating feature lists.
            part_index: Current chunk index.
        """
        if not self.features_output or not features_accumulator:
            return

        features_array: dict[str, Any] = {}
        for k, v_list in features_accumulator.items():
            features_array[k] = pa.concat_arrays(v_list)

        self._inject_dataloader_column(selection_name, features_array)
        self.features_output.write_table(selection_name, features_array, part_index)

    @staticmethod
    def _concatenate_accumulator(accumulator: dict[str, list[Any]]) -> dict[str, Any]:
        """Concatenate lists of arrays into a single dict of arrays."""
        return {k: pa.concat_arrays(v) for k, v in accumulator.items()}

    @staticmethod
    def _inject_path_prefixes(selection: DataSelection, metrics_processors: list[DatametricProcessor]) -> None:
        """Build per-column path prefix map from selection's sample_path config and inject into processors."""
        prefix_map: dict[str, str] = {}
        for entry in getattr(selection, "sample_path", []):
            col = entry.get("column")
            if col and entry.get("prefix"):
                prefix_map[col] = entry["prefix"]
        for metric in metrics_processors:
            metric.current_path_prefix = prefix_map

    @staticmethod
    def _clear_path_prefixes(metrics_processors: list[DatametricProcessor]) -> None:
        """Clear per-selection path prefix state from processors."""
        for metric in metrics_processors:
            if hasattr(metric, "current_path_prefix"):
                del metric.current_path_prefix

    def _compute_batches_metrics(
        self, selection_name: str, selection: DataSelection, metrics_processors: list[DatametricProcessor]
    ) -> dict[str, Any]:
        """Process all batches to compute intermediate statistics and features.

        Memory Management:
        - Batch-level statistics (`batch_metrics`) are accumulated in lists
          and concatenated once the selection is complete.
        - Per-sample features are also accumulated in memory before being
          passed to the OutputWriter.
        - NOTE: For large datasets, accumulation can lead to high memory
          usage. Future versions will implement disk-flushing (chunking).

        Args:
            selection_name: Name of the current data selection.
            selection: The selection iterator.
            metrics_processors: List of processors to apply to each batch.

        Returns:
            Dictionary of concatenated intermediate statistics arrays.
        """
        self._inject_path_prefixes(selection, metrics_processors)

        batch_metrics_accumulator: dict[str, list[Any]] = {}
        features_accumulator: dict[str, list[Any]] = {}
        feature_array_size = 0
        part_index = 0

        compute_max_memory = getattr(self, "compute_max_memory", None)
        memory_threshold = self._parse_memory_string(compute_max_memory) if compute_max_memory else 512 * 1024 * 1024

        dataloader_iter = (
            tqdm(selection, desc="batches", position=1, leave=False, total=selection.get_nb_batches())
            if self.progress_bar
            else selection
        )

        for batch in dataloader_iter:
            logger.debug(f"[DEBUG] _compute_batches_metrics: {selection_name} batch columns = {batch.schema.names}")
            batch_features, batch_metrics = self._process_batch(batch, metrics_processors)

            for k, v in batch_metrics.items():
                if k not in batch_metrics_accumulator:
                    batch_metrics_accumulator[k] = []
                batch_metrics_accumulator[k].append(v)

            feature_array_size = self._accumulate_source_features(batch, features_accumulator, feature_array_size)
            feature_array_size = self._accumulate_generated_features(
                batch, batch_features, batch_metrics, features_accumulator, feature_array_size
            )
            part_index = self._maybe_flush_features(
                selection_name, features_accumulator, feature_array_size, part_index, memory_threshold
            )
            if part_index > 0:
                feature_array_size = 0

        batches_metrics_array = self._concatenate_accumulator(batch_metrics_accumulator)
        self._write_remaining_features(selection_name, features_accumulator, part_index)
        self._clear_path_prefixes(metrics_processors)

        return batches_metrics_array

    def _parse_memory_string(self, memory_str: str) -> int:
        """Parse memory string (e.g., "2GB", "500MB") to bytes.

        Args:
            memory_str: Memory string to parse.

        Returns:
            Memory in bytes.
        """
        memory_str = memory_str.strip().upper()
        if memory_str.endswith("GB"):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1024)
        elif memory_str.endswith("B"):
            return int(float(memory_str[:-1]))
        else:
            # Assume it's in bytes
            return int(memory_str)

    def _inject_dataloader_column(self, selection_name: str, features: dict[str, Any]) -> None:
        """Inject the dataloader column into a features dict when configured.

        Adds the selection name as a column so the output parquet contains a
        ``dataloader`` column identifying which dataset each row originates from.

        Args:
            selection_name: Name of the current data selection (dataloader name).
            features: Mutable dict of column_name -> pa.Array to inject into.
        """
        if not self.features_output:
            return
        if not getattr(self.features_output, "add_dataloader_column", False):
            return

        col = self.features_output.dataloader_column_name
        if not features:
            return

        sample = next(iter(features.values()))
        features[col] = pa.array([selection_name] * len(sample))
