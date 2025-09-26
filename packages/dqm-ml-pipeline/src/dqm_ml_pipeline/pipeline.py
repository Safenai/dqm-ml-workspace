import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from .dataloaders import dqml_dataloaders_registry as dataloaders_registry
from .metrics import dqml_metrics_registry as metrics_registry
from .outputwriter import dqml_outputs_registry as outputs_registry

logger = logging.getLogger(__name__)

class DatasetPipeline:
    """
    Main class for processing datasets through a configurable pipeline
    of data loaders, metrics processors, and output writers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pipeline with a given configuration.

        Args:
            config: Dictionary containing:
                - dataloaders (dict)
                - metrics_processor (dict)
                - outputs (dict)
        """
        if not config:
            raise ValueError("Pipeline requires a configuration dictionary.")

        # Load data loaders
        if 'dataloaders' not in config or not isinstance(config['dataloaders'], dict):
            raise ValueError("'dataloaders' must be provided as a dictionary")
        self.dataloaders = self._init_components(config['dataloaders'], dataloaders_registry, "dataloader")

        # Load metrics
        if 'metrics_processor' not in config or not isinstance(config['metrics_processor'], dict):
            raise ValueError("'metrics_processor' must be provided as a dictionary")
        self.metrics = self._init_components(config['metrics_processor'], metrics_registry, "metric")

        # Determine needed input/generated columns
        self.needed_input_columns = []
        self.generated_features = []
        self.generated_metrics = []        
        for metric in self.metrics.values():
            self.needed_input_columns.extend(metric.needed_columns())
            self.generated_features.extend(metric.generated_features())
            self.generated_metrics.extend(metric.generated_metrics())

        # Deduplicate columns
        self.needed_input_columns = list(dict.fromkeys(self.needed_input_columns))
        self.generated_features = list(dict.fromkeys(self.generated_features))
        self.generated_metrics = list(dict.fromkeys(self.generated_metrics))


        # Load output writers
        if 'outputs' not in config or not isinstance(config['outputs'], dict):
            raise ValueError("'outputs' must be provided as a dictionary")
        
        self.metrics_output = None
        self.features_output = None
        for key, output_config in config['outputs'].items():
            if output_config['type'] not in outputs_registry:
                raise ValueError(f"Output '{key}' must have a valid 'type' in {list(outputs_registry.keys())}")
            writer = outputs_registry[output_config['type']](name=key, config=output_config)
            if key == "metrics":
                self.metrics_output = writer
            elif key == "features":
                self.features_output = writer
            else:
                raise ValueError(f"Unsupported output key '{key}'. Only 'features' and 'metrics' are allowed.")

        # Ensure output columns are included in needed input columns
        if self.features_output:
            for col in self.features_output.columns:
                if col not in self.generated_features:
                    logger.info(f"Adding required output column '{col}' to input columns")
                    self.needed_input_columns.insert(0, col)

    def _init_components(self, config_dict: Dict[str, Any], registry: Dict[str, Any], component_name: str) -> Dict[str, Any]:
        """Initialize pipeline components from a registry."""
        components = {}
        for key, comp_config in config_dict.items():
            if 'type' not in comp_config:
                raise ValueError(f"Configuration for {component_name} '{key}' must contain 'type'")
            comp_type = comp_config['type']
            if comp_type not in registry:
                raise ValueError(f"{component_name.capitalize()} '{key}' has invalid type '{comp_type}'")
            components[key] = registry[comp_type](name=key, config=comp_config)
        return components

    def get_ordered_metrics(self) -> List[Any]:
        """Return the ordered list of metrics processors."""
        # TODO: Implement proper ordering 
        return list(self.metrics.values())

    def run(self) -> None:
        """
        Execute the dataset processing pipeline.
        """
        # TODO: Check key uniqueness between batch, feature and metrics
        # TODO: Check with needed input order of metric computation
        metrics_processors = self.get_ordered_metrics()
        columns_list = self.needed_input_columns

        for dataloader_name, dataloader in self.dataloaders.items():
            logger.info(f"Processing dataloader '{dataloader_name}'")

            # TODO : We can add a progress bar here
            # TODO : wa can check available columns in the dataloader and check with needed columns

            metrics_array, features_array = {}, {}
            # TODO : increment if multiple tables
            # we compute the fefature size to enable partial writing of features            
            feature_array_size = 0
            part_index = 0

            for batch in dataloader.iterate(columns_list):
                batch_features, batch_metrics = {}, {}

                # Compute features and batch-level metrics
                for metric in metrics_processors:
                    logger.debug(f"Processing metric {metric.__class__.__name__} for dataloader {dataloader_name}")
                    batch_features |= metric.compute_features(batch, prev_features=batch_features)
                    batch_metrics |= metric.compute_batch_metric(batch_features)

                # Merge batch metrics
                # TODO : remove the change provide by check
                for k, v in batch_metrics.items():
                    if k in metrics_array:
                        # For histogram metrics, we need to sum, not concatenate
                         metrics_array[k] = pa.concat_arrays([metrics_array[k], v])
                    else:
                        metrics_array[k] = v

                # Copy required columns from source dataset
                for i, col_name in enumerate(batch.column_names):

                    # If we do not generate features we can skip it
                    if self.features_output == None:
                        continue

                    # If the feature is only use for internal computation, and not serialize in outputs, we skip it
                    if col_name not in self.features_output.columns:
                        continue

                    col_data = batch.column(i)
                    features_array[col_name] = pa.concat_arrays([features_array[col_name], col_data]) if col_name in features_array else col_data
                    feature_array_size += col_data.get_total_buffer_size()

                # Merge features for output
                for k, v in batch_features.items():

                     # If we do not generate features we can skip it
                    if self.features_output == None:
                        continue
                    # If the feature is also in the batch_metrics, no need to duplicate
                    if k not in self.features_output.columns or k in batch_metrics:
                        continue

                    features_array[k] = pa.concat_arrays([features_array[k], v]) if k in features_array else v
                    feature_array_size += v.get_total_buffer_size()
                
                # TODO: If feature_array_size > memory_limit, write features to disk and reset features_array
                # self.features_output(features_array, metrics_array)
                # TODO: In metrics we can have more data => handle in a dedicated function

            # Write features to disk
            # TODO: If too big parquet, save arrays, and start a new parquet file
            # If we have several feature to store
            if features_array:
                self.features_output.write_table(dataloader_name, features_array, part_index)

            # Compute dataset-level metrics
            dataset_metrics = {}
            for metric in metrics_processors:
                logger.debug(f"Processing metric computation {metric.__class__.__name__} for dataloader {dataloader_name}")
                dataset_metrics |= metric.compute(batch_metrics=metrics_array)
            
            # TODO : write to parquet, not implemented yet (done to test the pipeline with representativeness)
            if self.metrics_output and dataset_metrics:
                logger.debug(f"Writing metrics for dataloader {dataloader_name}")
                outputs = {}
                for key, values in dataset_metrics.items():
                    outputs[key] = pa.array([values])
                self.metrics_output.write_table(dataloader_name, outputs, part_index)
