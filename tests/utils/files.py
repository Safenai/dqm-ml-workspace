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


def write_path_list_to_parquet(
    path_list: list[Path], save_path: Path, class_list: list[str] | None = None, **columns: list[str]
) -> None:
    """Write list of file paths to a parquet file.

    Args:
        path_list: List of Path objects to write.
        save_path: Path where to save the parquet file.
        class_list: Optional list of class names corresponding to each path.
                    If provided, adds a 'class' column to the parquet.
        **columns: Additional named columns of equal length to include in the table.
    """
    path_str_list = [str(x) for x in path_list]
    path_array = pa.array(path_str_list)
    arrays = [path_array]
    names = ["image_path"]

    if class_list is not None:
        arrays.append(pa.array(class_list))
        names.append("class")

    for col_name, col_values in columns.items():
        arrays.append(pa.array(col_values))
        names.append(col_name)

    path_table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(path_table, save_path)
