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
    def _build_dependency_graph(procs: list[DatametricProcessor]) -> list[set[int]]:
        """Build a dependency graph from a list of processors.

        Args:
            procs: List of metric processors.

        Returns:
            List of sets where dep_on[i] contains indices of processors
            that processor i depends on.
        """
        generated_by = DatasetJob._register_generated_columns(procs)
        dep_on: list[set[int]] = [set() for _ in procs]
        for i, p in enumerate(procs):
            for col in p.needed_columns():
                for gen_idx in generated_by.get(col, ()):
                    if gen_idx != i:
                        dep_on[i].add(gen_idx)
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

        for i, col_name in enumerate(batch.column_names):
            if col_name not in self.features_output.columns:
                continue
            col_data = batch.column(i)
            if col_name not in features_accumulator:
                features_accumulator[col_name] = []
            features_accumulator[col_name].append(col_data)
            feature_array_size += col_data.get_total_buffer_size()
        return feature_array_size

    def _accumulate_generated_features(
        self,
        batch_features: dict[str, Any],
        batch_metrics: dict[str, Any],
        features_accumulator: dict[str, list[Any]],
        feature_array_size: int,
    ) -> int:
        """Accumulate generated features into the features accumulator.

        Args:
            batch_features: Features generated by processors.
            batch_metrics: Metrics generated by processors.
            features_accumulator: Dict accumulating feature lists.
            feature_array_size: Current memory usage estimate.

        Returns:
            Updated feature_array_size.
        """
        if self.features_output is None:
            return feature_array_size

        for k, v in batch_features.items():
            if k not in self.features_output.columns or k in batch_metrics:
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
        batch_metrics_accumulator: dict[str, list[Any]] = {}
        features_accumulator: dict[str, list[Any]] = {}
        feature_array_size = 0
        part_index = 0
        memory_threshold = 512 * 1024 * 1024

        dataloader_iter = (
            tqdm(selection, desc="batches", position=1, leave=False, total=selection.get_nb_batches())
            if self.progress_bar
            else selection
        )

        for batch in dataloader_iter:
            batch_features, batch_metrics = self._process_batch(batch, metrics_processors)

            for k, v in batch_metrics.items():
                if k not in batch_metrics_accumulator:
                    batch_metrics_accumulator[k] = []
                batch_metrics_accumulator[k].append(v)

            feature_array_size = self._accumulate_source_features(batch, features_accumulator, feature_array_size)
            feature_array_size = self._accumulate_generated_features(
                batch_features, batch_metrics, features_accumulator, feature_array_size
            )
            part_index = self._maybe_flush_features(
                selection_name, features_accumulator, feature_array_size, part_index, memory_threshold
            )
            if part_index > 0:
                feature_array_size = 0

        # Finalize
        batches_metrics_array: dict[str, Any] = {}
        for k, v_list in batch_metrics_accumulator.items():
            batches_metrics_array[k] = pa.concat_arrays(v_list)

        self._write_remaining_features(selection_name, features_accumulator, part_index)
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
