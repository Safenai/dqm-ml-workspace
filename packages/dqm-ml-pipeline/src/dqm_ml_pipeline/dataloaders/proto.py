from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSelection(Protocol):
    """
    Protocol for a specific selection of data from a DataLoader.
    """

    name: str

    def bootstrap(self, columns_list: list[str]) -> None:
        """Initialize the selection with required columns."""
        ...

    def get_nb_batches(self) -> int:
        """Get the number of batches for this selection."""
        ...

    def __iter__(self) -> Any:
        """Iterate over the selection batches."""
        ...


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for Data Loaders.

    A DataLoader is responsible for discovering and creating DataSelections.
    """

    def get_selections(self) -> list[DataSelection]:
        """
        Discover and return the list of available selections for this loader.
        """
        ...
