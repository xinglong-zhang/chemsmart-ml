"""Plot saved predictions without reading training databases or loading models."""

import csv
import json
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL_MOL = 627.509474


def plot_results(workdir, split="test", output=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError(
            "Plotting requires matplotlib: install chemsmart-ml[plot]"
        ) from exc
    root = Path(workdir)
    with (root / "samples.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "reference_energy", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "samples.csv lacks split/reference_energy/prediction columns"
            )
        rows = [row for row in reader if row["split"] == split]
    if not rows:
        raise ValueError(f"No samples for split {split}")
    reference = np.array([float(row["reference_energy"]) for row in rows])
    prediction = np.array([float(row["prediction"]) for row in rows])
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("Predictions contain nonfinite values")
    error = (prediction - reference) * HARTREE_TO_KCAL_MOL
    stats = dict(
        split=split,
        n=len(rows),
        mae_kcal_mol=float(abs(error).mean()),
        rmse_kcal_mol=float(np.sqrt(np.mean(error**2))),
        bias_kcal_mol=float(error.mean()),
        p95_absolute_error_kcal_mol=float(np.percentile(abs(error), 95)),
    )
    out = Path(output) if output else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        out / f"{split}-results.png",
        out / f"{split}-results.pdf",
        out / f"{split}-statistics.json",
    ]
    if any(path.exists() for path in paths):
        raise ValueError("Plot files already exist; choose a new --output directory")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    try:
        axes[0].hexbin(reference, prediction, gridsize=65, mincnt=1, bins="log")
        lo, hi = (
            min(reference.min(), prediction.min()),
            max(reference.max(), prediction.max()),
        )
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
        fig.suptitle(f"{split.capitalize()} predictions | {len(rows):,} samples")
        for path in paths[:2]:
            fig.savefig(path, dpi=170)
    finally:
        plt.close(fig)
    paths[2].write_text(json.dumps(stats, indent=2))
    return dict(statistics=stats, files=[str(path.resolve()) for path in paths])


def plot_cross_validation(report, output_dir):
    """Render per-fold MAE without interpreting fold variability as uncertainty."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError(
            "Plotting requires matplotlib: install chemsmart-ml[plot]"
        ) from exc
    folds = report.get("folds", [])
    if len(folds) < 2:
        raise ValueError("Cross-validation report must contain at least two folds")
    train = np.asarray([fold["train_mae_kcal_mol"] for fold in folds], dtype=float)
    valid = np.asarray([fold["mae_kcal_mol"] for fold in folds], dtype=float)
    if not (
        np.isfinite(train).all()
        and np.isfinite(valid).all()
        and (train >= 0).all()
        and (valid >= 0).all()
    ):
        raise ValueError("Cross-validation errors must be finite and nonnegative")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "cross-validation.png", out / "cross-validation.pdf"]
    if any(path.exists() for path in paths):
        raise ValueError(
            "Cross-validation plot files already exist; choose a new --output directory"
        )
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    try:
        positions = np.arange(len(folds))
        width = 0.32
        for offset, values, label, color in [
            (-width / 2, train, "Training", "#9CC8E3"),
            (width / 2, valid, "Validation", "#8063A5"),
        ]:
            bars = ax.bar(
                positions + offset, values, width, color=color, label=label, zorder=3
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values],
                padding=4,
                fontsize=9,
                color="#28334c",
            )
        ax.set_xticks(positions, [f"Fold {fold['fold']}" for fold in folds])
        ax.set_ylim(0, max(train.max(), valid.max(), 0.01) * 1.28)
        ax.set(xlabel="Validation fold", ylabel="Mean absolute error (kcal/mol)")
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
        mean, std = valid.mean(), valid.std(ddof=1)
        fig.text(
            0.5,
            -0.055,
            f"Validation MAE: {mean:.3f} ± {std:.3f} kcal/mol (mean ± fold SD)",
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
