import io
from typing import Any, Dict, List, Optional

from  dqm_ml_core.data_processor import DatametricProcessor
from PIL import Image
import pyarrow as pa
import pyarrow.compute as pc


def py_arrow_loading(ctx, x):
    print("py_arrow_loading : ",type(x))

    return pa.array(x)

# TODO : to not load images 
# https://fosdem.org/2025/events/attachments/fosdem-2025-6096-apache-arrow-tensor-arrays-an-approach-for-storing-tensor-data/slides/238048/Data_Anal_mPPA2ij.pdf

class ImageLoaderProcessor(DatametricProcessor):

    def __init__(self, name, config = None):
        super().__init__(name, config)
        function_name = "py_arrow_loading"
        function_docs = {
            "summary": "Load images from a pa.column",
            "description":
                "Given image_bytes use PIL to load images"
        }
        input_types = {
        "raw_bytes" : pa.int64(),
        }
        output_type = pa.int64()

        print("register scalar function")
        pc.register_scalar_function(py_arrow_loading,
                                    function_name,
                                    function_docs,
                                    input_types,
                                    output_type)

    def compute_features(self, batch: pa.RecordBatch, prev_features : pa.Array = None) -> dict[str, pa.Array]:
        """
        Compute the features for a given batch.
        
        Args:
            batch: The input batch of data.
        
        Returns:
            A dictionary of computed features.
        """
        # Get input column name from config
        input_columns = self.config.get('input_columns', ['image_bytes'])
        if not input_columns:
            return {}
            
        input_col = input_columns[0]  # Take first input column
        if input_col not in batch.schema.names:
            return {}
            
        col = batch.column(input_col)
        
        # Generate features according to output_columns mapping
        output_mapping = self.config.get('output_columns', {})
        features = {}
        
        if 'raw_image' in output_mapping:
            # raw_image maps to the image data
            features[output_mapping['raw_image']] = col
            
        if 'size' in output_mapping:
            # size maps to image size (placeholder implementation)
            batch_size = len(col)
            image_size = pa.array([batch_size] * batch_size, type=pa.int64())
            features[output_mapping['size']] = image_size
        
        return features