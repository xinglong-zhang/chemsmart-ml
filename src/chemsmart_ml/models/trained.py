"""Portable trained-model artifact (``model.json``)."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from chemsmart_ml.features.builder import FeatureBuilder, FeatureScaler
from chemsmart_ml.features.morgan import MorganFingerprint
from chemsmart_ml.features.soap import SOAPDescriptor
from chemsmart_ml.models.delta import DeltaModel

logger = logging.getLogger(__name__)


class TrainedModel:
    """
    Feature definition, scaler and fitted delta model of one training run.

    The JSON artifact holds only data (no pickles), so loading it never
    executes code. Predicted energy = low-level energy + correction.

    Attributes:
        features (FeatureBuilder): Feature definition.
        scaler (FeatureScaler): Training-set standardisation.
        delta_model (DeltaModel): Fitted correction.
        metadata (dict): Split, seed, plot flag and database paths.
    """

    FORMAT_VERSION = 1
    TARGET = "total_energy"
    UNIT = "Hartree"
    METADATA_KEYS = ("split", "seed", "plot", "source", "reference")

    def __init__(
        self,
        features: FeatureBuilder,
        scaler: FeatureScaler,
        delta_model: DeltaModel,
        metadata: Optional[dict] = None,
    ):
        self.features = features
        self.scaler = scaler
        self.delta_model = delta_model
        self.metadata = dict(metadata or {})

    @classmethod
    def from_dict(cls, data: dict) -> "TrainedModel":
        """Rebuild from a mapping, rejecting unsupported formats."""
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported model format")
        delta_model = DeltaModel.from_artifact(data)
        fingerprint = None
        if data.get("fingerprint"):
            fingerprint = MorganFingerprint.from_settings(data["fingerprint"])
        soap = None
        if data.get("soap"):
            soap = SOAPDescriptor.from_settings(data["soap"])
        features = FeatureBuilder(
            data["elements"],
            FeatureBuilder.features_from_electronic(data["electronic"]),
            fingerprint=fingerprint,
            soap=soap,
        )
        scaler = FeatureScaler(data["mean"], data["scale"])
        metadata = {k: data[k] for k in cls.METADATA_KEYS if k in data}
        return cls(features, scaler, delta_model, metadata)

    @classmethod
    def from_file(cls, filepath) -> "TrainedModel":
        return cls.from_dict(json.loads(Path(filepath).read_text()))

    def to_dict(self) -> dict:
        fields = self.delta_model.to_dict()
        fingerprint = self.features.fingerprint
        soap = self.features.soap
        return dict(
            format_version=self.FORMAT_VERSION,
            model=self.delta_model.artifact_name,
            alpha=fields["alpha"],
            target=self.TARGET,
            unit=self.UNIT,
            elements=self.features.elements,
            electronic=self.features.electronic,
            fingerprint=fingerprint.settings if fingerprint else None,
            mean=self.scaler.mean.tolist(),
            scale=self.scaler.scale.tolist(),
            coefficients=fields["coefficients"],
            tree=fields["tree"],
            nonlinear=fields["nonlinear"],
            soap=soap.settings if soap else None,
            **self.metadata,
        )

    def write(self, filepath):
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))

    def correction(self, x: np.ndarray) -> np.ndarray:
        """Predicted reference-minus-low-level energy (Hartree)."""
        return self.delta_model.predict(self.scaler.design_matrix(x))
