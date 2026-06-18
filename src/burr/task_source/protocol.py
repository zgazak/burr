"""
Formal protocol for task sources.

All task source managers (flats, calsats, photometry, coverage, lunar)
must implement this protocol.
"""

from typing import Protocol, runtime_checkable

from burr.models.observation import ObservationRequest
from burr.models.site import LightingCondition


@runtime_checkable
class TaskSource(Protocol):
    """Protocol that all task source managers must implement."""

    @property
    def name(self) -> str:
        """Short identifier for this task source (e.g. 'photometry', 'coverage')."""
        ...

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks."""
        ...

    def generate_request(
        self, task_id: str, binning: int = 1
    ) -> ObservationRequest | None:
        """Generate the next observation request.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor.

        Returns
        -------
        ObservationRequest | None
            A request if one is available, None otherwise.
        """
        ...

    def is_collection_complete(self) -> bool:
        """Check if this source has completed its collection goals."""
        ...

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully.

        Parameters
        ----------
        request : ObservationRequest
            The completed request.
        duration_seconds : float
            Wall-clock time the request took.
        """
        ...

    def on_request_failed(
        self, request: ObservationRequest, error: str
    ) -> None:
        """Called when an observation request fails.

        Parameters
        ----------
        request : ObservationRequest
            The failed request.
        error : str
            Description of the failure.
        """
        ...
