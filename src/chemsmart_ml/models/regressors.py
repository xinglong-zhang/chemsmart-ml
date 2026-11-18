"""
Portable regressors exported as JSON, without loading executable pickles.

Each regressor is fitted with scikit-learn, exported to plain parameters,
and predicts with NumPy alone. The export is verified against the
scikit-learn predictions immediately after fitting.

This module contains:
- PortableRegressor: Abstract fit/predict/serialise interface
- GradientBoostingTrees: Histogram gradient boosting as JSON trees
- NeuralNetwork: Two-hidden-layer ReLU multilayer perceptron
- RBFKernelRidge: Kernel ridge regression with an RBF kernel
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PortableRegressor(ABC):
    """
    Abstract regressor with a JSON-safe exported form.

    Attributes:
        sklearn_version (str): scikit-learn version used for fitting.
    """

    INSTALL_HINT = "Install chemsmart-ml[nonlinear] for nonlinear models"

    def __init__(self):
        self.sklearn_version = None

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "PortableRegressor":
        """Rebuild a fitted regressor from ``to_dict`` output."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "PortableRegressor":
        """Fit with scikit-learn and store the exported parameters."""

    @abstractmethod
    def predict(self, x) -> np.ndarray:
        """Predict with the exported parameters only."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-serialisable fitted parameters."""

    @classmethod
    def _import_sklearn(cls):
        try:
            import sklearn
        except ImportError as exc:
            raise ValueError(cls.INSTALL_HINT) from exc
        return sklearn

    def _verify_export(self, x, expected, tolerance, message):
        if not np.allclose(
            self.predict(x), expected, rtol=tolerance, atol=tolerance
        ):
            raise ValueError(message)


class GradientBoostingTrees(PortableRegressor):
    """
    Histogram gradient boosting exported as arrays of tree nodes.

    Each node is stored as [feature, threshold, left, right, value,
    is_leaf]. Categorical splits are not supported.

    Attributes:
        max_iter (int): Number of boosting stages.
        learning_rate (float): Shrinkage applied to each stage.
        max_leaf_nodes (int): Maximum leaves per tree.
        seed (int): Random state.
        baseline (float): Initial prediction.
        trees (list): Exported trees.
    """

    INSTALL_HINT = (
        "Tree training requires scikit-learn; install chemsmart-ml[trees]"
    )
    ARTIFACT_NAME = "hist-gradient-boosting-delta"
    VERIFY_SAMPLES = 1024

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        seed: int = 42,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed
        self.baseline = None
        self.trees = None

    @classmethod
    def from_dict(cls, data: dict) -> "GradientBoostingTrees":
        model = cls(
            data["max_iter"], data["learning_rate"], data["max_leaf_nodes"]
        )
        model.baseline = data["baseline"]
        model.trees = data["trees"]
        model.sklearn_version = data.get("scikit_learn_version")
        return model

    def fit(self, x, y) -> "GradientBoostingTrees":
        sklearn = self._import_sklearn()
        from sklearn.ensemble import HistGradientBoostingRegressor

        if (
            self.max_iter < 1
            or self.max_leaf_nodes < 2
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Invalid tree parameters")
        estimator = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.seed,
            early_stopping=False,
        )
        estimator.fit(x, y)
        self.trees = [
            self._export_nodes(stage[0].nodes)
            for stage in estimator._predictors
        ]
        self.baseline = float(estimator._baseline_prediction[0, 0])
        self.sklearn_version = sklearn.__version__
        check = x[: self.VERIFY_SAMPLES]
        self._verify_export(
            check,
            estimator.predict(check),
            1e-10,
            "Tree export does not match scikit-learn predictions",
        )
        return self

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Tree inputs must be finite")
        result = np.full(len(x), float(self.baseline))
        for tree in self.trees:
            nodes = np.asarray(tree, dtype=float)
            ids = np.zeros(len(x), dtype=int)
            while True:
                active = nodes[ids, 5] == 0
                if not active.any():
                    break
                ix = np.flatnonzero(active)
                selected = nodes[ids[ix]]
                features = selected[:, 0].astype(int)
                go_left = x[ix, features] <= selected[:, 1]
                ids[ix] = np.where(
                    go_left, selected[:, 2], selected[:, 3]
                ).astype(int)
            result += nodes[ids, 4]
        return result

    def to_dict(self) -> dict:
        return dict(
            model=self.ARTIFACT_NAME,
            baseline=self.baseline,
            trees=self.trees,
            scikit_learn_version=self.sklearn_version,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
        )

    @staticmethod
    def _export_nodes(nodes) -> list:
        if np.any(nodes["is_categorical"]):
            raise ValueError(
                "Categorical splits are not supported in JSON export"
            )
        return [
            [
                int(n["feature_idx"]),
                float(n["num_threshold"]),
                int(n["left"]),
                int(n["right"]),
                float(n["value"]),
                bool(n["is_leaf"]),
            ]
            for n in nodes
        ]


class NeuralNetwork(PortableRegressor):
    """
    Two-hidden-layer ReLU network trained on standardised targets.

    Attributes:
        hidden_size (int): Width of both hidden layers.
        max_iter (int): Maximum optimiser epochs.
        learning_rate (float): Initial Adam learning rate.
        seed (int): Random state.
        coefs (list): Layer weight matrices.
        intercepts (list): Layer bias vectors.
        target_mean (float): Training target mean.
        target_scale (float): Training target standard deviation.
        iterations (int): Epochs actually run.
    """

    KIND = "mlp"
    VERIFY_SAMPLES = 128

    def __init__(
        self,
        hidden_size: int = 128,
        max_iter: int = 300,
        learning_rate: float = 0.001,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed
        self.coefs = None
        self.intercepts = None
        self.target_mean = None
        self.target_scale = None
        self.iterations = None

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralNetwork":
        model = cls(
            data["hidden_size"], data["max_iter"], data["learning_rate"]
        )
        model.coefs = data["coefs"]
        model.intercepts = data["intercepts"]
        model.target_mean = data["target_mean"]
        model.target_scale = data["target_scale"]
        model.iterations = data["iterations"]
        model.sklearn_version = data.get("sklearn_version")
        return model

    def fit(self, x, y) -> "NeuralNetwork":
        sklearn = self._import_sklearn()
        from sklearn.neural_network import MLPRegressor

        # Scale residual targets on the training split for stable fitting.
        center, scale = float(np.mean(y)), float(np.std(y))
        if scale == 0:
            scale = 1.0
        estimator = MLPRegressor(
            hidden_layer_sizes=(self.hidden_size, self.hidden_size),
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            random_state=self.seed,
            early_stopping=False,
        )
        estimator.fit(x, (y - center) / scale)
        self.coefs = [a.tolist() for a in estimator.coefs_]
        self.intercepts = [a.tolist() for a in estimator.intercepts_]
        self.target_mean = center
        self.target_scale = scale
        self.iterations = int(estimator.n_iter_)
        self.sklearn_version = sklearn.__version__
        check = x[: self.VERIFY_SAMPLES]
        self._verify_export(
            check,
            estimator.predict(check) * scale + center,
            1e-8,
            "Nonlinear model export verification failed",
        )
        return self

    def predict(self, x) -> np.ndarray:
        z = np.asarray(x, dtype=float)
        for i, (coef, bias) in enumerate(zip(self.coefs, self.intercepts)):
            z = z @ np.asarray(coef) + np.asarray(bias)
            if i < len(self.coefs) - 1:
                z = np.maximum(z, 0)
        return z.ravel() * self.target_scale + self.target_mean

    def to_dict(self) -> dict:
        return dict(
            kind=self.KIND,
            coefs=self.coefs,
            intercepts=self.intercepts,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            hidden_size=self.hidden_size,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            sklearn_version=self.sklearn_version,
        )


class RBFKernelRidge(PortableRegressor):
    """
    Kernel ridge regression with a Gaussian (RBF) kernel.

    Training memory scales quadratically with the number of samples,
    so fitting is refused above ``max_samples``.

    Attributes:
        alpha (float): Ridge regularisation strength.
        gamma (float): RBF kernel coefficient.
        max_samples (int): Maximum number of training samples.
        support_vectors (list): Training inputs.
        dual_coef (list): Dual coefficients.
    """

    KIND = "kernel-ridge"
    VERIFY_SAMPLES = 128
    PREDICTION_BLOCK = 512

    def __init__(
        self,
        alpha: float = 0.001,
        gamma: float = 0.01,
        max_samples: int = 2000,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_samples = max_samples
        self.support_vectors = None
        self.dual_coef = None

    @classmethod
    def from_dict(cls, data: dict) -> "RBFKernelRidge":
        model = cls(data["alpha"], data["gamma"], data["max_samples"])
        model.support_vectors = data["support_vectors"]
        model.dual_coef = data["dual_coef"]
        model.sklearn_version = data.get("sklearn_version")
        return model

    def fit(self, x, y) -> "RBFKernelRidge":
        sklearn = self._import_sklearn()
        from sklearn.kernel_ridge import KernelRidge

        if self.max_samples < 1 or len(x) > self.max_samples:
            raise ValueError(
                f"Kernel ridge has {len(x)} training samples; limit is "
                f"{self.max_samples}. Use a smaller validated dataset; "
                "kernel memory scales quadratically."
            )
        estimator = KernelRidge(
            alpha=self.alpha, kernel="rbf", gamma=self.gamma
        )
        estimator.fit(x, y)
        self.support_vectors = x.tolist()
        self.dual_coef = estimator.dual_coef_.tolist()
        self.sklearn_version = sklearn.__version__
        check = x[: self.VERIFY_SAMPLES]
        self._verify_export(
            check,
            estimator.predict(check),
            1e-8,
            "Nonlinear model export verification failed",
        )
        return self

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        vectors = np.asarray(self.support_vectors)
        dual = np.asarray(self.dual_coef)
        # Bound the temporary kernel matrix during inference.
        values = []
        for start in range(0, len(x), self.PREDICTION_BLOCK):
            block = x[start : start + self.PREDICTION_BLOCK]
            distance = np.maximum(
                np.sum(block * block, axis=1)[:, None]
                + np.sum(vectors * vectors, axis=1)[None, :]
                - 2 * block @ vectors.T,
                0,
            )
            values.append(np.exp(-self.gamma * distance) @ dual)
        return np.concatenate(values)

    def to_dict(self) -> dict:
        return dict(
            kind=self.KIND,
            support_vectors=self.support_vectors,
            dual_coef=self.dual_coef,
            alpha=self.alpha,
            gamma=self.gamma,
            max_samples=self.max_samples,
            sklearn_version=self.sklearn_version,
        )
