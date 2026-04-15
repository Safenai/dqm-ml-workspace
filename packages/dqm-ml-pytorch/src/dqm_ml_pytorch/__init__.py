"""DQM ML PyTorch package for deep learning-based data quality metrics.

This package provides metric processors that use PyTorch models for
computing image embeddings and domain gap metrics.

Classes:
    ImageEmbeddingProcessor: Extracts image embeddings using pre-trained CNNs.
    DomainGapProcessor: Computes statistical distances between datasets.
"""

from dqm_ml_pytorch.domain_gap import DomainGapProcessor
from dqm_ml_pytorch.image_embedding import ImageEmbeddingProcessor

__all__ = ["DomainGapProcessor", "ImageEmbeddingProcessor"]
