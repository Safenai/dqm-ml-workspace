"""Protocol definitions for data loaders and selections.

This module contains the DataLoader and DataSelection protocol classes
that define the interface for data loading implementations.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSelection(Protocol):
    """
    Protocol for a specific subset of data discovered by a DataLoader.

    A DataSelection represents a concrete set of samples (e.g., a
    specific folder, a filtered view of a database, or a single file)
    and provides an iterator over data batches.
    """

    name: str

    def bootstrap(self, columns_list: list[str]) -> None:
        """Perform initial setup for the selection before iteration.

        Args:
            columns_list: List of column names to load.
        """

    def get_nb_batches(self) -> int:
        """
        Return the estimated number of batches in this selection.

        Used primarily for progress bar estimation.
        """

    def __iter__(self) -> Any:
        """
        Iterate over the selection, yielding pyarrow.RecordBatch objects.
        """


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for Data Loader factories.

    A DataLoader is responsible for scanning a source (disk, DB, S3) and
    discovering available DataSelections based on its configuration.
    """

    def get_selections(self) -> list[DataSelection]:
        """
        Discover and return the list of available selections for this loader.

        Returns:
            A list of initialized DataSelection instances.
        """
