"""Dataset job orchestrator for end-to-end data quality assessment.

This module contains the DatasetJob class that orchestrates the complete
pipeline: data loading, metric computation, and result persistence.
"""

import itertools
import logging
from typing import Any

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from dqm_ml_core.api.data_processor import DatametricProcessor
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
    ) -> None:
        """
        Initialize the pipeline components.

        Args:
            dataloaders: Map of initialized DataLoader instances.
            metrics: Map of initialized DatametricProcessor instances.
            features_output: Optional writer for persisting per-sample features.
            progress_bar: Whether to display execution progress in the terminal.
        """
        # We initialize loaded pluging elements
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.features_output = features_output
        self.progress_bar = progress_bar

        # Determine needed input/generated columns
        self.needed_input_columns: list[str] = []
        self.generated_features: list[str] = []
        self.generated_metrics: list[str] = []
        for metric in self.metrics.values():
            self.needed_input_columns.extend(metric.needed_columns())
            self.generated_features.extend(metric.generated_features())
            self.generated_metrics.extend(metric.generated_metrics())

        # Deduplicate columns
        self.needed_input_columns = list(dict.fromkeys(self.needed_input_columns))
        self.generated_features = list(dict.fromkeys(self.generated_features))
        self.generated_metrics = list(dict.fromkeys(self.generated_metrics))

        # Ensure output columns are included in needed input columns
        if self.features_output:
            for col in self.features_output.columns:
                if col not in self.generated_features:
                    logger.info(f"Adding required output column '{col}' to input columns")
                    self.needed_input_columns.insert(0, col)

        logger.info(
            f"DQM job pipeline initialized will process "
            f"{len(self.dataloaders)} dataloaders, "
            f"{len(self.metrics)} metrics processors, "
            f"outputting features to "
            f"'{self.features_output.name if self.features_output else 'None'}' "
        )

    @staticmethod
    def _build_dependency_graph(procs: list[DatametricProcessor]) -> list[set[int]]:
        """Build a dependency graph between metric processors.

        For each processor, collect the indices of processors that it depends on
        (i.e. producers of the columns it needs).

        Args:
            procs: List of metric processors.

        Returns:
            List of sets where ``dep_on[i]`` contains indices of processors
            that processor ``i`` depends on.
        """
        generated_by: dict[str, set[int]] = {}
        for i, p in enumerate(procs):
            for col in p.generated_features():
                generated_by.setdefault(col, set()).add(i)
            if hasattr(p, "generated_columns"):
                for col in p.generated_columns():
                    generated_by.setdefault(col, set()).add(i)

        dep_on: list[set[int]] = [set() for _ in procs]
        for i, p in enumerate(procs):
            for col in p.needed_columns():
                for gen_idx in generated_by.get(col, ()):
                    if gen_idx != i:
                        dep_on[i].add(gen_idx)
        return dep_on

    @staticmethod
    def _topological_sort(procs: list[DatametricProcessor], dep_on: list[set[int]]) -> list[DatametricProcessor]:
        """Return processors in topological order using Kahn's algorithm.

        Args:
            procs: List of metric processors.
            dep_on: Dependency graph built by ``_build_dependency_graph``.

        Returns:
            Processors ordered so that producers come before consumers.
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
        # TODO: Check with needed input order of metric computation
        metrics_processors = self.get_ordered_metrics()

        columns_list = self.needed_input_columns

        # Discover all selections
        all_selections: list[DataSelection] = []
        for loader in self.dataloaders.values():
            all_selections.extend(loader.get_selections())

        dataselection_metrics_list = {}

        job_iter = tqdm(all_selections, desc="selection", position=0) if self.progress_bar else all_selections  # noqa: E501

        # TODO : add as a specific command line argument
        self.describe(all_selections)

        for selection in job_iter:
            selection_name = selection.name
            logger.info(f"Processing selection '{selection_name}'")

            selection.bootstrap(columns_list)

            # Compute features and metrics for all batches
            batches_metrics_array = self._compute_batches_metrics(selection_name, selection, metrics_processors)

            # Compute dataset-level metrics
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

            dataselection_metrics_list[selection_name] = dataset_metrics

        # If we have to compute delta metrics
        delta_metrics_table = self._compute_delta_metrics(metrics_processors, dataselection_metrics_list)

        # Flush any accumulated features (single-output mode)
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
        # Use lists for O(1) appending, then concat once at the end.
        batch_metrics_accumulator: dict[str, list[Any]] = {}
        features_accumulator: dict[str, list[Any]] = {}

        # Track memory size for potential chunking
        feature_array_size = 0
        part_index = 0
        memory_threshold = 512 * 1024 * 1024  # 512MB threshold for flushing features

        dataloader_iter = (
            tqdm(selection, desc="batches", position=1, leave=False, total=selection.get_nb_batches())
            if self.progress_bar
            else selection
        )

        for batch in dataloader_iter:
            batch_features: dict[str, Any] = {}
            batch_metrics: dict[str, Any] = {}

            # Compute features and batch-level metrics
            for metric in metrics_processors:
                batch_features.update(metric.compute_features(batch, prev_features=batch_features))
                batch_metrics.update(metric.compute_batch_metric(batch_features))
                if logging.getLogger().level == logging.DEBUG:
                    m_keys, m_features = list(batch_metrics.keys()), list(batch_features.keys())
                    logger.debug(f"{metric.name} - Available batch_metrics  {m_keys} - features {m_features}")

            #  Accumulate batch metrics
            for k, v in batch_metrics.items():
                if k not in batch_metrics_accumulator:
                    batch_metrics_accumulator[k] = []
                batch_metrics_accumulator[k].append(v)

            # Accumulate features from source dataset
            for i, col_name in enumerate(batch.column_names):
                if self.features_output is None:
                    continue
                if col_name not in self.features_output.columns:
                    continue

                col_data = batch.column(i)
                if col_name not in features_accumulator:
                    features_accumulator[col_name] = []
                features_accumulator[col_name].append(col_data)
                feature_array_size += col_data.get_total_buffer_size()

            # Accumulate generated features
            for k, v in batch_features.items():
                if self.features_output is None:
                    continue
                # Avoid duplication if feature is also a metric or not required
                if k not in self.features_output.columns or k in batch_metrics:
                    continue

                if k not in features_accumulator:
                    features_accumulator[k] = []
                features_accumulator[k].append(v)
                feature_array_size += v.get_total_buffer_size()

            # Flush features to disk if memory threshold reached
            if feature_array_size > memory_threshold and self.features_output:
                logger.info(
                    f"Memory threshold reached ({feature_array_size / 1024**2:.1f}MB). Flushing chunk {part_index}"
                )
                features_chunk: dict[str, Any] = {}
                for k, v_list in features_accumulator.items():
                    features_chunk[k] = pa.concat_arrays(v_list)

                self._inject_dataloader_column(selection_name, features_chunk)
                self.features_output.write_table(selection_name, features_chunk, part_index)

                # Reset features accumulator
                features_accumulator = {}
                feature_array_size = 0
                part_index += 1

        # Concatenate all accumulated arrays
        batches_metrics_array: dict[str, Any] = {}
        for k, v_list in batch_metrics_accumulator.items():
            batches_metrics_array[k] = pa.concat_arrays(v_list)

        features_array: dict[str, Any] = {}
        if features_accumulator:
            for k, v_list in features_accumulator.items():
                features_array[k] = pa.concat_arrays(v_list)

        # Write remaining features to disk
        if self.features_output and features_array:
            self._inject_dataloader_column(selection_name, features_array)
            self.features_output.write_table(selection_name, features_array, part_index)

        return batches_metrics_array

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
