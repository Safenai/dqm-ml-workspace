"""File utility functions for DQM-ML tests.

This module provides helper functions for file operations.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def get_files_list(path: Path, pattern: str = "*.parquet") -> list[str]:
    """Get list of files matching a pattern in a directory.

    Args:
        path: Directory path to search.
        pattern: Glob pattern to match files.

    Returns:
        List of file paths as strings.
    """
    files = sorted(Path(path).glob(pattern))
    path_list = [str(x) for x in files]
    return path_list


def write_path_list_to_parquet(path_list: list[Path], save_path: Path) -> None:
    """Write list of file paths to a parquet file.

    Args:
        path_list: List of Path objects to write.
        save_path: Path where to save the parquet file.
    """
    path_str_list = [str(x) for x in path_list]
    path_array = pa.array(path_str_list)
    path_table = pa.Table.from_arrays([path_array], names=["image_path"])
    pq.write_table(path_table, save_path)
