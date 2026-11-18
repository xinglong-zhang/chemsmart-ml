"""Populate the Morgan fingerprint cache without labels or training."""

import logging
from pathlib import Path

from chemsmart_ml.features.morgan import MorganFingerprint
from chemsmart_ml.io.database import SourceDatabase
from chemsmart_ml.io.manifest import ValidationManifest
from chemsmart_ml.workflows.base import MLWorkflow

logger = logging.getLogger(__name__)


class FingerprintPreparation(MLWorkflow):
    """
    Compute reusable Morgan fingerprints for a source database.

    Attributes:
        filename (Path): Source database.
        fingerprint (MorganFingerprint): Fingerprint and cache to fill.
        manifest (Path): Optional validation manifest restricting samples.
        output (Path): New directory for the cache report.
    """

    ACTION = "prepare"

    def __init__(
        self, filename, output, fingerprint: MorganFingerprint, manifest=None
    ):
        super().__init__(output)
        self.filename = Path(filename)
        self.fingerprint = fingerprint
        self.manifest = Path(manifest) if manifest else None

    @classmethod
    def from_options(
        cls,
        filename,
        output,
        morgan_radius=2,
        morgan_size=2048,
        morgan_chirality=False,
        cache=None,
        manifest=None,
    ) -> "FingerprintPreparation":
        fingerprint = MorganFingerprint(
            morgan_radius, morgan_size, morgan_chirality, cache=cache
        )
        return cls(filename, output, fingerprint, manifest)

    def run(self) -> dict:
        self._create_output_directory()
        database = SourceDatabase(self.filename, include_smiles=True)
        records = database.records
        if self.manifest:
            manifest = ValidationManifest.from_file(self.manifest)
            manifest.verify(database)
            records = manifest.select_source_records(records)
        _, report = self._compute_descriptor(
            self.fingerprint, list(records.values())
        )
        return report
