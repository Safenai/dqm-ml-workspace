"""Plot utility functions for DQM-ML tests.

This module provides helper functions for generating plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from tests.utils.files import get_files_list


def plot_histograms(
    output_path: Path,
    columns: list[str],
    parquets_path: Path,
    parquet_pattern: str,
) -> None:
    """Generate histogram plots for columns in parquet files.

    Args:
        output_path: Directory to save generated plots.
        columns: List of column names to plot.
        parquets_path: Directory containing parquet files.
        parquet_pattern: Pattern to match parquet files.
    """
    path_list = get_files_list(parquets_path, pattern=parquet_pattern)

    for parquet_path in path_list:
        parquet_name = Path(parquet_path).stem

        table = pq.read_table(parquet_path)
        for column in columns:
            dist = table[column].to_numpy()

            fig, ax = plt.subplots()

            ax.hist(dist, bins=100)

            plot_name = f"hist_{parquet_name}_{column}.png"

            plt.title(f"Histogram of {column} in {parquet_name}")
            plt.savefig(f"{output_path}/{plot_name}")
            plt.close(fig)
