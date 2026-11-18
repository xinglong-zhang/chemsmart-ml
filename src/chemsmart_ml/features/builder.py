"""
Source-side model inputs and their standardisation.

This module contains:
- FeatureBuilder: Element counts, xTB energy, electronic properties and
  optional cached descriptors
- FeatureScaler: Training-set standardisation and design matrices
"""

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from chemsmart_ml.features.base import CachedDescriptor
from chemsmart_ml.features.morgan import MorganFingerprint
from chemsmart_ml.features.soap import SOAPDescriptor
from chemsmart_ml.io.database import MolecularRecord

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Assemble the feature matrix from source records only.

    Columns are ordered: element counts, low-level total energy
    (Hartree), selected electronic properties, Morgan bits, then SOAP.
    The first ``len(elements) + 1`` columns form the composition/energy
    baseline used by residual models.

    Attributes:
        elements (list[str]): Element basis from the training set.
        features (str): 'composition-energy', 'orbitals' or 'electronic'.
        fingerprint (MorganFingerprint): Optional Morgan fingerprint.
        soap (SOAPDescriptor): Optional SOAP descriptor.
    """

    ELECTRONIC_KEYS = {
        "composition-energy": (),
        "orbitals": ("homo", "lumo"),
        "electronic": ("homo", "lumo", "dipole"),
    }
    # Value stored as ``electronic`` in model and manifest artifacts.
    ELECTRONIC_ARTIFACT_VALUES = {
        "composition-energy": False,
        "orbitals": "orbitals",
        "electronic": True,
    }

    def __init__(
        self,
        elements: Sequence[str],
        features: str = "composition-energy",
        fingerprint: Optional[MorganFingerprint] = None,
        soap: Optional[SOAPDescriptor] = None,
    ):
        if features not in self.ELECTRONIC_KEYS:
            raise ValueError(f"Unsupported feature set {features!r}")
        self.elements = list(elements)
        self.features = features
        self.fingerprint = fingerprint
        self.soap = soap

    @classmethod
    def features_from_electronic(cls, electronic) -> str:
        """Map a stored ``electronic`` artifact value to a feature set."""
        for name, value in cls.ELECTRONIC_ARTIFACT_VALUES.items():
            if value == electronic and type(value) is type(electronic):
                return name
        raise ValueError(f"Unsupported electronic feature mode {electronic!r}")

    @property
    def electronic(self):
        """Feature mode as stored in model artifacts."""
        return self.ELECTRONIC_ARTIFACT_VALUES[self.features]

    @property
    def electronic_keys(self) -> Tuple[str, ...]:
        return self.ELECTRONIC_KEYS[self.features]

    @property
    def base_size(self) -> int:
        """Number of composition/energy baseline columns."""
        return len(self.elements) + 1

    @property
    def descriptors(self) -> List[CachedDescriptor]:
        return [d for d in (self.fingerprint, self.soap) if d is not None]

    def base_vector(self, record: MolecularRecord) -> List[float]:
        """Element counts, energy and electronic properties of a record."""
        names = list(self.elements) + ["energy"] + list(self.electronic_keys)
        values = [record.atoms.count(z) for z in self.elements]
        values += [record.energy]
        values += [getattr(record, key) for key in self.electronic_keys]
        missing = [
            name
            for name, value in zip(names, values)
            if value is None or not np.isfinite(value)
        ]
        if missing:
            raise ValueError(
                f"Missing/nonfinite input features {missing} for record "
                f"{record.record_id}; inspect feature-availability.json"
            )
        if set(record.atoms) - set(self.elements):
            raise ValueError(
                "Prediction contains elements absent from training"
            )
        return values

    def base_matrix(self, records: Sequence[MolecularRecord]) -> np.ndarray:
        return np.array([self.base_vector(r) for r in records], dtype=float)

    def matrix(
        self,
        records: Sequence[MolecularRecord],
        descriptor_vectors: Sequence[np.ndarray] = (),
    ) -> np.ndarray:
        """
        Full feature matrix of records.

        Args:
            records: Source records.
            descriptor_vectors: Precomputed rows of each descriptor in
                ``descriptors`` order, aligned with ``records``.
        """
        base = self.base_matrix(records)
        if len(descriptor_vectors) != len(self.descriptors):
            raise ValueError("One vector block is required per descriptor")
        return np.column_stack([base, *descriptor_vectors])

    def descriptor_vectors(
        self, records: Sequence[MolecularRecord]
    ) -> List[np.ndarray]:
        """Compute each configured descriptor in column order."""
        return [d.compute(records)[0] for d in self.descriptors]


class FeatureScaler:
    """
    Standardise features with training-set mean and standard deviation.

    Constant columns keep a unit scale.

    Attributes:
        mean (np.ndarray): Per-column mean, or None before fitting.
        scale (np.ndarray): Per-column scale, or None before fitting.
    """

    def __init__(self, mean=None, scale=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.scale = None if scale is None else np.asarray(scale, dtype=float)

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1
        return self

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Standardised features with a leading intercept column."""
        if self.mean is None:
            raise ValueError("FeatureScaler must be fitted before use")
        return np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
