"""Image loading mixin for processors that load images from storage.

This module provides the ImageLoadingMixin class that handles loading
images from local filesystem paths or S3, resolving path prefixes
set by the job for each data selection.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class ImageLoadingMixin:
    """Mixin for processors that load images from S3 or local paths.

    Expects the host class to provide:
    - ``self.s3_fs``: An S3 filesystem instance (or ``None``).
    - ``self.current_path_prefix``: A dict mapping column names to path
      prefixes, set by the job per data selection.
    """

    s3_fs: Any = None

    def _current_image_prefix(self, column: str | None = None) -> str | None:
        """Return the path prefix for the given column.

        Reads from ``self.current_path_prefix``, a dict set by the job
        mapping column names to path prefixes.

        Args:
            column: The input column name. Falls back to the first input
                column if ``None``.

        Returns:
            The path prefix string, or ``None``.
        """
        prefix_map: dict[str, str] = getattr(self, "current_path_prefix", {})
        if column is not None:
            return prefix_map.get(column)
        cols: list[str] = getattr(self, "input_columns", [])
        return prefix_map.get(cols[0]) if cols else None

    def _open_s3_image(self, prefix: str, path: str) -> Image.Image:
        """Open a PIL Image from S3, reading into memory and copying.

        Args:
            prefix: The path prefix within the bucket.
            path: The relative image path.

        Returns:
            A PIL Image object.
        """
        bucket = os.getenv("S3_BUCKET_NAME", "")
        s3_path = f"{bucket}/{prefix}/{path}"
        with self.s3_fs.open_input_stream(s3_path) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return img.copy()

    def _open_image_from_path(self, path: str, column: str | None = None) -> Image.Image | None:
        """Open a PIL Image from a string path (S3 or local filesystem).

        Resolves relative paths by looking up the prefix for the configured
        input column from ``self.current_path_prefix`` (set by the job for
        each selection).

        Args:
            path: File path or relative path.
            column: The input column name (used to resolve the path prefix).

        Returns:
            A PIL Image object, or ``None`` if the file does not exist.
        """
        prefix = self._current_image_prefix(column)

        if self.s3_fs:
            return self._open_s3_image(prefix or "", path)

        img_path = Path(prefix) / path if prefix else Path(path)
        if not img_path.is_file():
            logger.warning(f"Path does not exist: {img_path}")
            return None
        return Image.open(img_path)
