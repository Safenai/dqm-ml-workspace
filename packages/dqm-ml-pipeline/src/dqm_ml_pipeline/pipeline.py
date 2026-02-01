import itertools
import logging
from typing import Any

import pyarrow as pa
from tqdm import tqdm

from dqm_ml_core.api.data_processor import DatametricProcessor
from dqm_ml_pipeline.dataloaders import DataLoader
from dqm_ml_pipeline.outputwriter import OutputWriter

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    Main class for processing datasets through a configurable pipeline.

    This class orchestrates the data loading, metric computation, and result writing
    processes. It supports dynamically loaded plugins for data loaders, metrics, and
    output writers.
    """

    def __init__(self,
                 dataloaders: dict[str, DataLoader],
                 metrics: dict[str, DatametricProcessor],
                 features_output: OutputWriter | None, progress_bar: bool = True) -> None:
        """
        Initialize the pipeline with a given configuration.

        Args:
            config: Dictionary containing the pipeline configuration:
                - dataloaders: Dict of data loader configurations.
                - metrics_processor: Dict of metric processor configurations.
                - outputs: Dict of output writer configurations.
                - compute_delta: Boolean, whether to compute delta metrics between datasets.
                - progress_bar: Boolean, whether to show progress bars.

        Raises:
            ValueError: If the configuration is missing or invalid.
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
            f"DQM job pipeline initiazed will process {len(self.dataloaders)} dataloaders, "  # noqa: E501
            f"{len(self.metrics)} metrics processors, "
            f"outputting features to '{self.features_output.name if self.features_output else 'None'}' "
        )

    def get_ordered_metrics(self) -> list[DatametricProcessor]:
        """
        Return the ordered list of metrics processors.

        Returns:
            List of ordered DatametricProcessor instances.
        """
        # TODO: Implement proper ordering based on dependencies
        return list(self.metrics.values())

    def describe(self) -> None:
        """
        Log a description of the pipeline configuration.
        """
        logger.info(f"Executing dqm-ml-job on {len(self.dataloaders)}, using {len(self.metrics)} metrics ")
        for dataloader_name, dataloader in self.dataloaders.items():
            logger.info(f"  Dataloader: {dataloader_name} -> {dataloader}")

        for metric_name, metric in self.metrics.items():
            logger.info(f"  Metric: {metric_name} -> {metric}")
            logger.info(f"    Needed columns: {metric.needed_columns()}")
            logger.info(f"    Generated features: {metric.generated_features()}")
            logger.info(f"    Generated metrics: {metric.generated_metrics()}")

    def run(self) -> tuple[dict[Any, dict[str, Any]], dict[str, Any] | None]:
        """
        Execute the dataset processing pipeline.

        Iterates through configured data loaders, computes metrics, and writes outputs.
        """

        # TODO: Check with needed input order of metric computation
        metrics_processors = self.get_ordered_metrics()

        columns_list = self.needed_input_columns

        dataselection_metrics_list = {}

        if self.progress_bar:
            pipeline_iter = tqdm(self.dataloaders.items(), desc="loader", position=0)
        else:
            pipeline_iter = self.dataloaders.items()

        # TODO : add as a specific command line argument
        self.describe()

        for dataloader_name, dataloader in pipeline_iter:
            logger.info(f"Processing dataloader '{dataloader_name}'")

            # TODO add filter for parquet columns if supported by dataloader
            dataloader.bootstrap(columns_list)

            # Compute features and metrics for all batches
            batches_metrics_array = self._compute_batches_metrics(dataloader_name, dataloader, metrics_processors)

            # Compute dataset-level metrics
            dataset_metrics: dict[str, Any] = {}
            for metric in metrics_processors:
                if logging.getLogger().level == logging.DEBUG:
                    logger.debug(f"Metric computation {metric.__class__.__name__} for dataselection {dataloader_name}")
                dataset_metrics.update(metric.compute(batch_metrics=batches_metrics_array))
                if logging.getLogger().level == logging.DEBUG:
                    logger.debug(f"Available metrics  {list(dataset_metrics.keys())}")

            dataselection_metrics_list[dataloader_name] = dataset_metrics

        # If we have to compute delta metrics
        delta_metrics_table = self._compute_delta_metrics(metrics_processors, dataselection_metrics_list)

        return dataselection_metrics_list, delta_metrics_table

    def _compute_delta_metrics(self, metrics_processors: list[DatametricProcessor],
                               dataselection_metrics_list: dict[str, dict[str, pa.Array]]) -> dict[str, Any] | None:
        """
        Compute delta metrics between all combinations of dataselections.
        Args:
            metrics_processors: List of metric processors to use for delta computation.
            dataselection_metrics_list: A dictionary mapping dataselection names to their computed metrics.
        Returns:
            A dictionary containing delta metrics for each combination of dataselections.
        """

        selection_combinaisons = itertools.combinations(dataselection_metrics_list, 2)

        delta_metrics_table = None
        for combinaison in selection_combinaisons:
            src_metrics = dataselection_metrics_list[combinaison[0]]
            target_metrics = dataselection_metrics_list[combinaison[1]]

            for metric in metrics_processors:
                delta_metrics = metric.compute_delta(src_metrics, target_metrics)
                # TODO : check format of classical metrics / delta metrics for combinaison of format
                if len(delta_metrics) == 0:
                    continue

                if delta_metrics_table is None:
                    delta_metrics_table = delta_metrics
                    delta_metrics_table["selection_source"] = pa.array([combinaison[0]])
                    delta_metrics_table["selection_target"] = pa.array([combinaison[1]])
                else:
                    for m_name, value in delta_metrics.items():
                        delta_metrics_table[m_name] = pa.concat_arrays([delta_metrics_table[m_name], pa.array([value])])

                    delta_metrics_table["selection_source"] = pa.concat_arrays([delta_metrics_table["selection_source"], pa.array([combinaison[0]])])  # noqa: E501
                    delta_metrics_table["selection_target"] = pa.concat_arrays([delta_metrics_table["selection_target"], pa.array([combinaison[1]])])  # noqa: E501
                    logger.debug(f"Writing delta metrics for dataloader {'_'.join(combinaison)}")

        return delta_metrics_table

    def _compute_batches_metrics(self, dataloader_name: str, dataloader: DataLoader,
                                 metrics_processors: list[DatametricProcessor]) -> dict[str, Any]:
        """
        Compute metrics and features for all batches in a dataloader.

        This method optimizes performance by accumulating batch results in lists
        and concatenating them once at the end, avoiding O(N^2) complexity.

        Args:
            dataloader_name: Name of the dataloader.
            dataloader: The dataloader instance.
            metrics_processors: List of metric processors to apply.

        Returns:
            A tuple containing:
                - batches_metrics_array: Dictionary of computed batch metrics (concatenated).
        """
        # Use lists for O(1) appending, then concat once at the end.
        batch_metrics_accumulator: dict[str, list[Any]] = {}
        features_accumulator: dict[str, list[Any]] = {}

        # Track memory size for potential chunking (not fully implemented yet)
        feature_array_size = 0
        part_index = 0

        if self.progress_bar:
            dataloader_iter = tqdm(
                dataloader, total=dataloader.get_nb_batches(), desc="batches", position=1, leave=False
            )
        else:
            dataloader_iter = dataloader

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

            # TODO: If feature_array_size > memory_limit, write features to disk and reset accumulators

        # Concatenate all accumulated arrays
        batches_metrics_array: dict[str, Any] = {}
        for k, v_list in batch_metrics_accumulator.items():
            batches_metrics_array[k] = pa.concat_arrays(v_list)

        features_array: dict[str, Any] = {}
        for k, v_list in features_accumulator.items():
            features_array[k] = pa.concat_arrays(v_list)

        # Write features to disk
        # TODO: If too big parquet, save arrays, and start a new parquet file (chunking)
        if self.features_output and features_array:
            self.features_output.write_table(dataloader_name, features_array, part_index)

        return batches_metrics_array
