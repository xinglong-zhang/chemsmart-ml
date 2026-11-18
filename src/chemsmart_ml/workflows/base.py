"""Base class for command workflows writing auditable artifacts."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

logger = logging.getLogger(__name__)


class MLWorkflow(ABC):
    """
    Abstract workflow run by one ``chemsmart-ml`` subcommand.

    Concrete subclasses register under ``ACTION``. Directory outputs
    must be new; artifacts are never silently overwritten.

    Attributes:
        output (Path): Output directory, or output file for predictions.
    """

    ACTION: Optional[str] = None
    _REGISTRY: Dict[str, Type["MLWorkflow"]] = {}

    def __init__(self, output):
        self.output = Path(output)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.ACTION is not None:
            MLWorkflow._REGISTRY[cls.ACTION] = cls

    def __repr__(self):
        return f"{self.__class__.__qualname__}<output={self.output}>"

    @classmethod
    def from_action(cls, action: str, options: dict) -> "MLWorkflow":
        """Create the workflow registered for a subcommand."""
        try:
            workflow_cls = cls._REGISTRY[action]
        except KeyError:
            raise ValueError(f"Unknown action {action!r}") from None
        return workflow_cls.from_options(**options)

    @classmethod
    def from_options(cls, **options) -> "MLWorkflow":
        """Create from command options; override to build settings."""
        return cls(**options)

    @abstractmethod
    def run(self) -> dict:
        """Run the workflow and return its JSON-serialisable summary."""

    def _create_output_directory(self):
        self.output.mkdir(parents=True, exist_ok=False)

    def _write_json(self, filename: str, data):
        (self.output / filename).write_text(json.dumps(data, indent=2))

    def _compute_descriptor(self, descriptor, records):
        """Compute descriptor vectors, recording a report or the error."""
        try:
            vectors, report = descriptor.compute(records)
        except ValueError as exc:
            self._write_json(
                descriptor.ERROR_FILENAME,
                dict(error=str(exc), settings=descriptor.settings),
            )
            raise
        self._write_json(descriptor.REPORT_FILENAME, report)
        return vectors, report
