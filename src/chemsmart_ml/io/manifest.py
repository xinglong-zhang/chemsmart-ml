"""Explicit, reusable manifests of samples that passed input screening."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from chemsmart_ml.io.database import CHEMSMARTDatabase, MolecularRecord

logger = logging.getLogger(__name__)


class ValidationManifest:
    """
    Screened sample selection tied to the exact database files it audited.

    Attributes:
        source (dict): Identity of the audited source database.
        reference (dict): Identity of the audited reference database.
        electronic (bool | str): Electronic feature mode that was checked.
        fingerprint (str): Fingerprint whose inputs were checked.
        samples (list[dict]): Accepted samples with record identifiers.
        filepath (Path): File the manifest was read from, if any.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        source: dict,
        reference: Optional[dict],
        electronic,
        fingerprint: str,
        samples: List[dict],
        filepath=None,
    ):
        self.source = source
        self.reference = reference
        self.electronic = electronic
        self.fingerprint = fingerprint
        self.samples = list(samples)
        self.filepath = Path(filepath).resolve() if filepath else None

    @classmethod
    def from_file(cls, filepath) -> "ValidationManifest":
        """Read a manifest written by ``DatasetValidation``."""
        data = json.loads(Path(filepath).read_text())
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported validation manifest format; rerun validate"
            )
        return cls(
            source=data.get("source"),
            reference=data.get("reference"),
            electronic=data.get("electronic"),
            fingerprint=data.get("fingerprint"),
            samples=data.get("samples", []),
            filepath=filepath,
        )

    @property
    def selected_samples(self) -> Dict[int, dict]:
        """Accepted samples keyed by zero-based NPZ index."""
        return {item["npz_index"]: item for item in self.samples}

    def to_dict(self) -> dict:
        return dict(
            format_version=self.FORMAT_VERSION,
            source=self.source,
            reference=self.reference,
            electronic=self.electronic,
            fingerprint=self.fingerprint,
            samples=self.samples,
        )

    def write(self, filepath):
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))

    def verify(
        self,
        source: CHEMSMARTDatabase,
        reference: Optional[CHEMSMARTDatabase] = None,
    ):
        """Reject databases changed since the manifest was written."""
        if self.source != source.identity:
            raise ValueError(
                "Validation manifest does not match the source database; "
                "rerun validate"
            )
        if reference is not None and self.reference != reference.identity:
            raise ValueError(
                "Validation manifest does not match the reference database; "
                "rerun validate"
            )

    def select_source_records(
        self, records: Dict[int, MolecularRecord]
    ) -> Dict[int, MolecularRecord]:
        """Keep only accepted source records, checking record identifiers."""
        selected = self.selected_samples
        kept = {key: row for key, row in records.items() if key in selected}
        if len(kept) != len(selected) or any(
            row.record_id != selected[key]["source_record"]
            for key, row in kept.items()
        ):
            raise ValueError("Source sample IDs differ from manifest")
        return kept
