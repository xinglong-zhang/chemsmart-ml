"""Validated settings for training runs."""

import copy
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from chemsmart_ml.features.builder import FeatureBuilder
from chemsmart_ml.models.delta import DeltaModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingSettings:
    """
    Complete, validated configuration of one training run.

    Settings are checked on construction, before any database is read.
    ``learning_rate`` defaults to 0.001 for 'mlp' and 0.1 otherwise.
    """

    filename: str
    reference: str
    output: str
    features: str = "composition-energy"
    model: str = "linear"
    alpha: float = 1.0
    seed: int = 42
    plot: bool = True
    fingerprint: str = "none"
    morgan_radius: int = 2
    morgan_size: int = 2048
    morgan_chirality: bool = False
    cache: Optional[str] = None
    manifest: Optional[str] = None
    train_fraction: float = 0.8
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    descriptor: str = "none"
    soap_r_cut: float = 5.0
    soap_n_max: int = 2
    soap_l_max: int = 2
    soap_sigma: float = 0.5
    soap_aggregation: str = "sum"
    soap_cache: Optional[str] = None
    max_iter: int = 100
    learning_rate: Optional[float] = None
    max_leaf_nodes: int = 31
    hidden_size: int = 128
    kernel_alpha: float = 0.001
    kernel_gamma: float = 0.01
    kernel_max_samples: int = 2000
    kernel_subsample: bool = False
    cv_folds: int = 0

    FINGERPRINTS = ("none", "morgan")
    DESCRIPTORS = ("none", "soap")
    CROSS_VALIDATION_MODELS = ("hist-gradient-boosting",)

    def __post_init__(self):
        if self.learning_rate is None:
            self.learning_rate = 0.001 if self.model == "mlp" else 0.1
        self.validate()

    def validate(self):
        """Raise ValueError for inconsistent or unsupported settings."""
        if self.cv_folds and (
            self.model not in self.CROSS_VALIDATION_MODELS or self.cv_folds < 2
        ):
            raise ValueError(
                "Cross-validation currently supports hist-gradient-boosting "
                "with at least two folds"
            )
        if (
            self.model not in DeltaModel.names()
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError("Invalid model or regularization strength")
        fractions = np.array(list(self.split_fractions.values()), dtype=float)
        if (
            not np.isfinite(fractions).all()
            or (fractions <= 0).any()
            or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-9)
        ):
            raise ValueError("Split fractions must be positive and sum to 1")
        if self.features not in FeatureBuilder.ELECTRONIC_KEYS:
            raise ValueError(f"Unsupported feature set {self.features!r}")
        if self.fingerprint not in self.FINGERPRINTS:
            raise ValueError("Unsupported fingerprint")
        if self.descriptor not in self.DESCRIPTORS:
            raise ValueError("Unsupported 3D descriptor")
        if self.plot:
            try:
                # Check before the expensive database read.
                __import__("matplotlib")
            except ImportError as exc:
                raise ValueError(
                    "Automatic plotting requires matplotlib; install "
                    "chemsmart-ml[plot] or use --no-plot"
                ) from exc

    @property
    def split_fractions(self) -> dict:
        return dict(
            train=self.train_fraction,
            valid=self.valid_fraction,
            test=self.test_fraction,
        )

    @property
    def source_path(self) -> str:
        return str(Path(self.filename).resolve())

    @property
    def reference_path(self) -> str:
        return str(Path(self.reference).resolve())

    def copy(self) -> "TrainingSettings":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)
