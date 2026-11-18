"""
Delta models predicting reference-minus-low-level energy corrections.

All models act on a standardised design matrix whose first column is
the intercept. Residual models first fit an unregularised least-squares
baseline on the composition/energy columns, which removes the extensive
elemental offset, and then learn a nonlinear correction of the residual.

This module contains:
- DeltaModel: Abstract base with a registry keyed by model name
- LinearDeltaModel, RidgeDeltaModel: Linear corrections
- ResidualDeltaModel: Baseline plus portable nonlinear regressor
- GradientBoostingDeltaModel, MLPDeltaModel, KernelRidgeDeltaModel
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

from chemsmart_ml.models.regressors import (
    GradientBoostingTrees,
    NeuralNetwork,
    PortableRegressor,
    RBFKernelRidge,
)

logger = logging.getLogger(__name__)


class DeltaModel(ABC):
    """
    Abstract energy-correction model.

    Concrete subclasses register under ``NAME``; the stored artifact
    name is ``<NAME>-delta``. The artifact fields ``alpha``,
    ``coefficients``, ``tree`` and ``nonlinear`` are provided by
    ``to_dict``, with None for fields a model does not use.
    """

    NAME: Optional[str] = None
    _REGISTRY: Dict[str, Type["DeltaModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.NAME is not None:
            DeltaModel._REGISTRY[cls.NAME] = cls

    @classmethod
    def names(cls):
        """Registered model names."""
        return list(cls._REGISTRY)

    @classmethod
    def from_settings(cls, settings, base_size: int) -> "DeltaModel":
        """
        Create an unfitted model from training settings.

        Args:
            settings (TrainingSettings): Final training settings.
            base_size (int): Number of baseline design columns,
                including the intercept.
        """
        try:
            model_cls = cls._REGISTRY[settings.model]
        except KeyError:
            raise ValueError(f"Unknown model {settings.model!r}") from None
        return model_cls._from_settings(settings, base_size)

    @classmethod
    def from_artifact(cls, artifact: dict) -> "DeltaModel":
        """Rebuild a fitted model from a ``model.json`` mapping."""
        name = artifact.get("model", "")
        model_cls = cls._REGISTRY.get(name.removesuffix("-delta"))
        if not name.endswith("-delta") or model_cls is None:
            raise ValueError("Unsupported model format")
        return model_cls._from_artifact(artifact)

    @property
    def artifact_name(self) -> str:
        return f"{self.NAME}-delta"

    @classmethod
    @abstractmethod
    def _from_settings(cls, settings, base_size: int) -> "DeltaModel":
        """Construct from settings; see ``from_settings``."""

    @classmethod
    @abstractmethod
    def _from_artifact(cls, artifact: dict) -> "DeltaModel":
        """Construct from an artifact; see ``from_artifact``."""

    @abstractmethod
    def fit(
        self, design: np.ndarray, delta: np.ndarray, rows: np.ndarray
    ) -> "DeltaModel":
        """
        Fit the correction.

        Args:
            design: Standardised design matrix of all samples.
            delta: Reference minus low-level energy of all samples.
            rows: Indices of the training samples.
        """

    @abstractmethod
    def predict(self, design: np.ndarray) -> np.ndarray:
        """Predicted correction for each design row."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Model-specific artifact fields."""


class LinearDeltaModel(DeltaModel):
    """
    Ordinary least-squares correction.

    Attributes:
        alpha (float): Regularisation strength, recorded for provenance;
            ordinary least squares does not use it.
        coefficients (np.ndarray): Fitted coefficients.
    """

    NAME = "linear"

    def __init__(self, alpha: Optional[float] = None, coefficients=None):
        self.alpha = alpha
        self.coefficients = (
            None if coefficients is None else np.asarray(coefficients)
        )

    @classmethod
    def _from_settings(cls, settings, base_size):
        return cls(alpha=settings.alpha)

    @classmethod
    def _from_artifact(cls, artifact):
        return cls(artifact.get("alpha"), artifact["coefficients"])

    def fit(self, design, delta, rows):
        self.coefficients = np.linalg.lstsq(
            design[rows], delta[rows], rcond=None
        )[0]
        return self

    def predict(self, design):
        return design @ self.coefficients

    def to_dict(self):
        return dict(
            alpha=self.alpha,
            coefficients=self.coefficients.tolist(),
            tree=None,
            nonlinear=None,
        )


class RidgeDeltaModel(LinearDeltaModel):
    """Ridge correction with an unpenalised intercept."""

    NAME = "ridge"
    CHUNK_SIZE = 2048

    def fit(self, design, delta, rows):
        if not self.alpha:
            return super().fit(design, delta, rows)
        # Accumulate normal equations blockwise; avoids an expensive
        # tall-matrix SVD and a full copy of the training matrix.
        gram = np.eye(design.shape[1]) * self.alpha
        gram[0, 0] = 0
        rhs = np.zeros(design.shape[1])
        for start in range(0, len(rows), self.CHUNK_SIZE):
            selected = rows[start : start + self.CHUNK_SIZE]
            block = design[selected]
            gram += block.T @ block
            rhs += block.T @ delta[selected]
        self.coefficients = np.linalg.solve(gram, rhs)
        return self


class ResidualDeltaModel(DeltaModel):
    """
    Least-squares baseline plus a portable nonlinear residual model.

    The regressor sees all standardised features (design columns after
    the intercept).

    Attributes:
        regressor (PortableRegressor): Residual model.
        base_size (int): Baseline design columns, including intercept.
        base_coefficients (np.ndarray): Fitted baseline coefficients.
    """

    ARTIFACT_KEY: Optional[str] = None
    REGRESSOR: Optional[Type[PortableRegressor]] = None

    def __init__(
        self,
        regressor: PortableRegressor,
        base_size: int,
        base_coefficients=None,
    ):
        self.regressor = regressor
        self.base_size = base_size
        self.base_coefficients = (
            None
            if base_coefficients is None
            else np.asarray(base_coefficients)
        )

    @classmethod
    def _from_artifact(cls, artifact):
        data = artifact[cls.ARTIFACT_KEY]
        coefficients = data["base_coefficients"]
        return cls(
            cls.REGRESSOR.from_dict(data), len(coefficients), coefficients
        )

    def fit(self, design, delta, rows):
        base = design[rows, : self.base_size]
        self.base_coefficients = np.linalg.lstsq(
            base, delta[rows], rcond=None
        )[0]
        residual = delta[rows] - base @ self.base_coefficients
        self.regressor.fit(design[rows, 1:], residual)
        return self

    def predict(self, design):
        baseline = design[:, : self.base_size] @ self.base_coefficients
        return baseline + self.regressor.predict(design[:, 1:])

    def to_dict(self):
        fields = dict(alpha=None, coefficients=None, tree=None, nonlinear=None)
        fields[self.ARTIFACT_KEY] = self._regressor_artifact()
        return fields

    def _regressor_artifact(self) -> dict:
        artifact = self.regressor.to_dict()
        artifact["base_coefficients"] = self.base_coefficients.tolist()
        return artifact


class GradientBoostingDeltaModel(ResidualDeltaModel):
    """Baseline plus histogram gradient-boosted trees."""

    NAME = "hist-gradient-boosting"
    ARTIFACT_KEY = "tree"
    REGRESSOR = GradientBoostingTrees

    @classmethod
    def _from_settings(cls, settings, base_size):
        regressor = GradientBoostingTrees(
            max_iter=settings.max_iter,
            learning_rate=settings.learning_rate,
            max_leaf_nodes=settings.max_leaf_nodes,
            seed=settings.seed,
        )
        return cls(regressor, base_size)


class NonlinearDeltaModel(ResidualDeltaModel):
    """
    Residual model stored under the ``nonlinear`` artifact key.

    Attributes:
        kernel_subsample (bool): Whether training samples were
            subsampled for a kernel model; None for other models.
    """

    ARTIFACT_KEY = "nonlinear"

    def __init__(
        self,
        regressor,
        base_size,
        base_coefficients=None,
        kernel_subsample: Optional[bool] = None,
    ):
        super().__init__(regressor, base_size, base_coefficients)
        self.kernel_subsample = kernel_subsample

    @classmethod
    def _from_artifact(cls, artifact):
        model = super()._from_artifact(artifact)
        model.kernel_subsample = artifact[cls.ARTIFACT_KEY].get(
            "kernel_subsample"
        )
        return model

    def _regressor_artifact(self):
        artifact = super()._regressor_artifact()
        artifact["kernel_subsample"] = self.kernel_subsample
        return artifact


class MLPDeltaModel(NonlinearDeltaModel):
    """Baseline plus a multilayer perceptron."""

    NAME = "mlp"
    REGRESSOR = NeuralNetwork

    @classmethod
    def _from_settings(cls, settings, base_size):
        regressor = NeuralNetwork(
            hidden_size=settings.hidden_size,
            max_iter=settings.max_iter,
            learning_rate=settings.learning_rate,
            seed=settings.seed,
        )
        return cls(regressor, base_size)


class KernelRidgeDeltaModel(NonlinearDeltaModel):
    """Baseline plus RBF kernel ridge regression."""

    NAME = "kernel-ridge"
    REGRESSOR = RBFKernelRidge

    @classmethod
    def _from_settings(cls, settings, base_size):
        regressor = RBFKernelRidge(
            alpha=settings.kernel_alpha,
            gamma=settings.kernel_gamma,
            max_samples=settings.kernel_max_samples,
        )
        return cls(
            regressor,
            base_size,
            kernel_subsample=bool(settings.kernel_subsample),
        )
