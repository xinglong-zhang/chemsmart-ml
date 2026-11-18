"""
Paired source/reference samples, molecular groups and data splits.

This module contains:
- SamplePair: One source record matched to one reference record
- PairedDataset: Matched samples with a pairing report and group IDs
- DataSplit: Group-aware train/validation/test labels
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from chemsmart_ml.io.database import CHEMSMARTDatabase, MolecularRecord
from chemsmart_ml.io.manifest import ValidationManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplePair:
    """
    One source record matched to one reference record.

    Attributes:
        index (int): Zero-based NPZ index shared by both records.
        source (MolecularRecord): Low-level record.
        reference (MolecularRecord): Reference record.
    """

    index: int
    source: MolecularRecord
    reference: MolecularRecord


class PairedDataset:
    """
    Matched source/reference samples and the report of how they paired.

    Group IDs are connected components over both endpoint molecule IDs,
    so neither a source nor a reference identity can cross splits. They
    are computed over all matched pairs before any manifest selection.

    Attributes:
        pairs (list[SamplePair]): Matched samples in NPZ index order.
        report (dict): Unmatched, rejected and selection counts.
        group_ids (list[str]): Molecular group of each pair.
    """

    def __init__(
        self,
        pairs: Iterable[SamplePair],
        report: dict,
        group_ids: Optional[List[str]] = None,
    ):
        self.pairs = list(pairs)
        self.report = report
        if group_ids is None:
            group_ids = self._connected_components(self.pairs)
        if len(group_ids) != len(self.pairs):
            raise ValueError("Each sample pair requires one group ID")
        self.group_ids = list(group_ids)

    @classmethod
    def from_records(
        cls,
        source: Dict[int, MolecularRecord],
        reference: Dict[int, MolecularRecord],
    ) -> "PairedDataset":
        """Pair records sharing an index, rejecting composition mismatches."""
        pairs, rejected = [], []
        for key in sorted(source.keys() & reference.keys()):
            a, b = source[key], reference[key]
            if not a.has_same_composition(b):
                rejected.append(
                    dict(index=key, reason="element composition mismatch")
                )
                continue
            pairs.append(SamplePair(key, a, b))
        report = dict(
            source_only=sorted(source.keys() - reference.keys()),
            reference_only=sorted(reference.keys() - source.keys()),
            rejected=rejected,
            matched=len(pairs),
            identity_changes=sum(
                p.source.molecule_id != p.reference.molecule_id for p in pairs
            ),
        )
        return cls(pairs, report)

    @classmethod
    def from_databases(
        cls, source: CHEMSMARTDatabase, reference: CHEMSMARTDatabase
    ) -> "PairedDataset":
        return cls.from_records(source.records, reference.records)

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    @property
    def source_records(self) -> List[MolecularRecord]:
        return [p.source for p in self.pairs]

    @property
    def source_energies(self) -> np.ndarray:
        """Low-level total energies in Hartree."""
        return np.array([p.source.energy for p in self.pairs])

    @property
    def reference_energies(self) -> np.ndarray:
        """Reference total energies in Hartree."""
        return np.array([p.reference.energy for p in self.pairs], dtype=float)

    @property
    def elements(self) -> List[str]:
        """Sorted chemical symbols present in the source structures."""
        return sorted({z for p in self.pairs for z in p.source.atoms})

    def finite_counts(self, names: Iterable[str]) -> Dict[str, int]:
        """Count source records with a finite value for each property."""
        return {
            name: sum(p.source.has_finite(name) for p in self.pairs)
            for name in names
        }

    def select(self, manifest: ValidationManifest) -> "PairedDataset":
        """Return the samples accepted by a validation manifest."""
        selected = manifest.selected_samples
        for p in self.pairs:
            item = selected.get(p.index)
            if item and (
                p.source.record_id != item["source_record"]
                or p.reference.record_id != item["reference_record"]
            ):
                raise ValueError("Sample IDs differ from validation manifest")
        kept = [i for i, p in enumerate(self.pairs) if p.index in selected]
        if len(kept) != len(selected):
            raise ValueError("Validation manifest contains unmatched samples")
        report = dict(
            self.report,
            selected=len(kept),
            validation_manifest=str(manifest.filepath),
            excluded_by_manifest=self.report["matched"] - len(kept),
        )
        return PairedDataset(
            [self.pairs[i] for i in kept],
            report,
            [self.group_ids[i] for i in kept],
        )

    @staticmethod
    def _connected_components(pairs: List[SamplePair]) -> List[str]:
        """Union source and reference molecule IDs into shared groups."""
        parent = {}

        def root(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in pairs:
            x, y = root(p.source.molecule_id), root(p.reference.molecule_id)
            parent[max(x, y)] = min(x, y)
        return [root(p.source.molecule_id) for p in pairs]


class DataSplit:
    """
    Group-aware assignment of samples to train/valid/test partitions.

    Attributes:
        labels (np.ndarray): Partition label of each sample. Samples
            excluded from training by subsampling are labelled 'unused'.
    """

    NAMES = ("train", "valid", "test")

    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype="<U6")

    @classmethod
    def from_groups(
        cls, group_ids: Iterable[str], fractions: Dict[str, float], seed: int
    ) -> "DataSplit":
        """
        Allocate whole molecular groups by a stable seeded hash.

        Args:
            group_ids: Molecular group of each sample.
            fractions (dict): Train/valid/test fractions summing to 1.
            seed (int): Seed mixed into the hash.
        """
        train, valid = fractions["train"], fractions["valid"]
        labels = []
        for group in group_ids:
            digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
            h = int(digest[:12], 16) / 16**12
            if h < train:
                labels.append("train")
            elif h < train + valid:
                labels.append("valid")
            else:
                labels.append("test")
        return cls(labels)

    def __len__(self):
        return len(self.labels)

    def mask(self, name: str) -> np.ndarray:
        return self.labels == name

    def count(self, name: str) -> int:
        return int(self.mask(name).sum())

    def check_nonempty(self):
        if any(not self.mask(name).any() for name in self.NAMES):
            raise ValueError(
                "Empty split; need more independent molecular groups"
            )

    def subsample_training(self, max_samples: int, seed: int):
        """Keep at most ``max_samples`` training samples; mark the rest."""
        candidates = np.flatnonzero(self.mask("train"))
        if len(candidates) <= max_samples:
            return
        selected = np.random.default_rng(seed).choice(
            candidates, size=max_samples, replace=False
        )
        self.labels[candidates] = "unused"
        self.labels[selected] = "train"
        logger.info(
            f"Subsampled training set from {len(candidates)} to "
            f"{max_samples} samples"
        )
