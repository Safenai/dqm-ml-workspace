from typing import Any, Dict, List, Optional

import pyarrow as pa

from dqm_ml_core.data_processor import DatametricProcessor


class ImageFeaturesProcessor(DatametricProcessor):

    def compute_features(self, batch: pa.RecordBatch, prev_features : pa.Array = None) -> dict[str, pa.Array]:
        """
        Compute the features for a given batch.
        
        Args:
            batch: The input batch of data.
        
        Returns:
            A dictionary of computed features.
        """
        return {}
