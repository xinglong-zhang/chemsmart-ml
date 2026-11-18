"""
Plots of saved results, without reading databases or loading models.

This module contains:
- MatplotlibPlotter: Shared backend selection and file safety
- PredictionResultsPlotter: Prediction/error plots for one split
- CrossValidationPlotter: Per-fold training/validation MAE
"""

import csv
import json
import logging
from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np

from chemsmart_ml.constants import HARTREE_TO_KCAL_MOL

logger = logging.getLogger(__name__)


class MatplotlibPlotter:
    """Base class for plots rendered with the non-interactive Agg backend."""

    @staticmethod
    def pyplot():
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ValueError(
                "Plotting requires matplotlib: install chemsmart-ml[plot]"
            ) from exc
        return plt

    @staticmethod
    def _ensure_new(paths: List[Path], message: str):
        """Refuse to overwrite existing plot files."""
        if any(path.exists() for path in paths):
            raise ValueError(message)


class PredictionResultsPlotter(MatplotlibPlotter):
    """
    Prediction-versus-reference and error plots from ``samples.csv``.

    Attributes:
        workdir (Path): Training output directory.
        split (str): Partition to plot ('train', 'valid' or 'test').
    """

    REQUIRED_COLUMNS = {"split", "reference_energy", "prediction"}

    def __init__(self, workdir, split: str = "test"):
        self.workdir = Path(workdir)
        self.split = split

    @cached_property
    def samples(self) -> List[dict]:
        """Rows of ``samples.csv`` belonging to the selected split."""
        with (self.workdir / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not self.REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError(
                    "samples.csv lacks split/reference_energy/prediction "
                    "columns"
                )
            rows = [row for row in reader if row["split"] == self.split]
        if not rows:
            raise ValueError(f"No samples for split {self.split}")
        return rows

    @property
    def reference(self) -> np.ndarray:
        """Reference energies (Hartree)."""
        return np.array([float(r["reference_energy"]) for r in self.samples])

    @property
    def prediction(self) -> np.ndarray:
        """Predicted energies (Hartree)."""
        return np.array([float(r["prediction"]) for r in self.samples])

    @property
    def statistics(self) -> dict:
        """Error statistics in kcal/mol."""
        error = self._error(self.reference, self.prediction)
        return dict(
            split=self.split,
            n=len(self.samples),
            mae_kcal_mol=float(abs(error).mean()),
            rmse_kcal_mol=float(np.sqrt(np.mean(error**2))),
            bias_kcal_mol=float(error.mean()),
            p95_absolute_error_kcal_mol=float(np.percentile(abs(error), 95)),
        )

    def plot(self, output=None) -> dict:
        """
        Write PNG/PDF plots and a statistics JSON.

        Args:
            output (str | Path, optional): Directory for plot files.
                Defaults to ``<workdir>/plots``.
        """
        plt = self.pyplot()
        reference, prediction = self.reference, self.prediction
        error = self._error(reference, prediction)
        stats = self.statistics
        out = Path(output) if output else self.workdir / "plots"
        out.mkdir(parents=True, exist_ok=True)
        paths = [
            out / f"{self.split}-results.png",
            out / f"{self.split}-results.pdf",
            out / f"{self.split}-statistics.json",
        ]
        self._ensure_new(
            paths, "Plot files already exist; choose a new --output directory"
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        try:
            axes[0].hexbin(
                reference, prediction, gridsize=65, mincnt=1, bins="log"
            )
            lo = min(reference.min(), prediction.min())
            hi = max(reference.max(), prediction.max())
            axes[0].plot([lo, hi], [lo, hi], color="coral")
            axes[0].set(
                xlabel="Reference DFT energy (Hartree)",
                ylabel="Predicted DFT energy (Hartree)",
                title="Prediction vs reference",
            )
            axes[1].hist(error, bins=80, color="steelblue")
            axes[1].axvline(0, color="coral")
            axes[1].set(
                xlabel="Prediction minus reference (kcal/mol)",
                ylabel="Samples",
                title=f"MAE = {stats['mae_kcal_mol']:.2f} kcal/mol",
            )
            axes[2].hexbin(reference, error, gridsize=60, mincnt=1, bins="log")
            axes[2].axhline(0, color="coral")
            axes[2].set(
                xlabel="Reference DFT energy (Hartree)",
                ylabel="Error (kcal/mol)",
                title="Errors across energy range",
            )
            fig.suptitle(
                f"{self.split.capitalize()} predictions | "
                f"{len(reference):,} samples"
            )
            for path in paths[:2]:
                fig.savefig(path, dpi=170)
        finally:
            plt.close(fig)
        paths[2].write_text(json.dumps(stats, indent=2))
        return dict(statistics=stats, files=[str(p.resolve()) for p in paths])

    @staticmethod
    def _error(reference, prediction) -> np.ndarray:
        """Prediction minus reference in kcal/mol, requiring finite input."""
        if not (
            np.isfinite(reference).all() and np.isfinite(prediction).all()
        ):
            raise ValueError("Predictions contain nonfinite values")
        return (prediction - reference) * HARTREE_TO_KCAL_MOL


class CrossValidationPlotter(MatplotlibPlotter):
    """
    Per-fold training and validation MAE from a cross-validation report.

    Fold variability is shown as spread, not as a prediction uncertainty.

    Attributes:
        report (dict): Contents of ``cross-validation.json``.
    """

    REPORT_FILENAME = "cross-validation.json"
    COLORS = {"Training": "#9CC8E3", "Validation": "#8063A5"}

    def __init__(self, report: dict):
        self.report = report

    @classmethod
    def from_file(cls, filepath) -> "CrossValidationPlotter":
        return cls(json.loads(Path(filepath).read_text()))

    @property
    def folds(self) -> List[dict]:
        folds = self.report.get("folds", [])
        if len(folds) < 2:
            raise ValueError(
                "Cross-validation report must contain at least two folds"
            )
        return folds

    @property
    def train_mae(self) -> np.ndarray:
        return self._fold_values("train_mae_kcal_mol")

    @property
    def validation_mae(self) -> np.ndarray:
        return self._fold_values("mae_kcal_mol")

    def plot(self, output_dir) -> dict:
        """Write PNG/PDF fold plots to ``output_dir``."""
        plt = self.pyplot()
        folds, train, valid = self.folds, self.train_mae, self.validation_mae
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / "cross-validation.png", out / "cross-validation.pdf"]
        self._ensure_new(
            paths,
            "Cross-validation plot files already exist; choose a new "
            "--output directory",
        )
        mean, std = valid.mean(), valid.std(ddof=1)
        fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
        try:
            positions = np.arange(len(folds))
            width = 0.32
            for offset, values, label in [
                (-width / 2, train, "Training"),
                (width / 2, valid, "Validation"),
            ]:
                bars = ax.bar(
                    positions + offset,
                    values,
                    width,
                    color=self.COLORS[label],
                    label=label,
                    zorder=3,
                )
                ax.bar_label(
                    bars,
                    labels=[f"{value:.3f}" for value in values],
                    padding=4,
                    fontsize=9,
                    color="#28334c",
                )
            ax.set_xticks(positions, [f"Fold {f['fold']}" for f in folds])
            ax.set_ylim(0, max(train.max(), valid.max(), 0.01) * 1.28)
            ax.set(
                xlabel="Validation fold",
                ylabel="Mean absolute error (kcal/mol)",
            )
            ax.set_title(
                "Grouped cross-validation",
                loc="left",
                fontsize=15,
                fontweight="bold",
                pad=18,
            )
            ax.grid(axis="y", color="#e0e3ee", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="upper right", ncol=2)
            fig.text(
                0.5,
                -0.055,
                f"Validation MAE: {mean:.3f} ± {std:.3f} kcal/mol "
                "(mean ± fold SD)",
                ha="center",
                fontsize=11,
                color="#4d3c6d",
            )
            for path in paths:
                fig.savefig(path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)
        return dict(
            files=[str(path.resolve()) for path in paths],
            validation_mae_mean_kcal_mol=float(mean),
            validation_mae_std_kcal_mol=float(std),
        )

    def _fold_values(self, key: str) -> np.ndarray:
        values = np.asarray([fold[key] for fold in self.folds], dtype=float)
        if not (np.isfinite(values).all() and (values >= 0).all()):
            raise ValueError(
                "Cross-validation errors must be finite and nonnegative"
            )
        return values
