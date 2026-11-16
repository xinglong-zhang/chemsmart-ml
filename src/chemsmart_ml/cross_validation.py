"""Grouped development-set cross-validation with an untouched holdout."""

import csv
import json
from pathlib import Path

import click
import numpy as np

from .tree_model import fit_tree, predict_tree

HARTREE_TO_KCAL_MOL = 627.509474


def cross_validate_tree(
    x,
    low,
    target,
    pairs,
    group_ids,
    labels,
    out,
    *,
    seed,
    n_splits,
    elements,
    max_iter,
    learning_rate,
    max_leaf_nodes,
    plot,
    model_settings,
):
    try:
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        raise ValueError("Cross-validation requires scikit-learn>=1.6") from exc
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds")
    out = Path(out)
    development = np.flatnonzero(labels != "test")
    holdout = np.flatnonzero(labels == "test")
    group_ids = np.asarray(group_ids)
    if not len(holdout) or len(np.unique(group_ids[development])) < n_splits:
        raise ValueError("Not enough groups or no held-out test samples")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(labels), -1, dtype=int)
    folds = []
    for fold, (local_train, local_valid) in enumerate(
        splitter.split(development, groups=group_ids[development]), start=1
    ):
        train = development[local_train]
        valid = development[local_valid]
        assignment[valid] = fold
        click.echo(
            f"Cross-validation fold {fold}/{n_splits}: {len(train)} training, {len(valid)} validation samples",
            err=True,
        )
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale == 0] = 1
        xt = (x[train] - mean) / scale
        xv = (x[valid] - mean) / scale
        base_size = len(elements) + 1
        train_base = np.column_stack([np.ones(len(train)), xt[:, :base_size]])
        valid_base = np.column_stack([np.ones(len(valid)), xv[:, :base_size]])
        delta = target[train] - low[train]
        coef = np.linalg.lstsq(train_base, delta, rcond=None)[0]
        model = fit_tree(
            xt, delta - train_base @ coef, seed, max_iter, learning_rate, max_leaf_nodes
        )
        training_error = (
            low[train] + train_base @ coef + predict_tree(xt, model)
        ) - target[train]
        error = (low[valid] + valid_base @ coef + predict_tree(xv, model)) - target[
            valid
        ]
        folds.append(
            dict(
                fold=fold,
                train_n=len(train),
                validation_n=len(valid),
                train_mae_kcal_mol=float(
                    np.abs(training_error).mean() * HARTREE_TO_KCAL_MOL
                ),
                mae_hartree=float(np.abs(error).mean()),
                mae_kcal_mol=float(np.abs(error).mean() * HARTREE_TO_KCAL_MOL),
                rmse_kcal_mol=float(np.sqrt(np.mean(error**2)) * HARTREE_TO_KCAL_MOL),
            )
        )
    if np.any(assignment[development] < 1) or np.any(assignment[holdout] != -1):
        raise ValueError("Invalid fold assignment")
    with (out / "cv-assignments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "npz_index",
                "idx",
                "source_record",
                "reference_record",
                "group_id",
                "partition",
                "validation_fold",
            ]
        )
        for i, (idx, a, b) in enumerate(pairs):
            w.writerow(
                [
                    idx,
                    idx + 1,
                    a["record_id"],
                    b["record_id"],
                    group_ids[i],
                    "holdout" if assignment[i] == -1 else "development",
                    "" if assignment[i] == -1 else int(assignment[i]),
                ]
            )
    vals = np.array([item["mae_kcal_mol"] for item in folds])
    train_vals = np.array([item["train_mae_kcal_mol"] for item in folds])
    report = dict(
        n_splits=n_splits,
        seed=seed,
        development_samples=len(development),
        held_out_test_samples=len(holdout),
        held_out_test_evaluated=False,
        model=model_settings,
        folds=folds,
        train_mae_mean_kcal_mol=float(train_vals.mean()),
        validation_mae_mean_kcal_mol=float(vals.mean()),
        validation_mae_std_kcal_mol=float(vals.std(ddof=1)),
    )
    (out / "cross-validation.json").write_text(json.dumps(report, indent=2))
    if plot:
        from .plotting import plot_cross_validation

        plot_cross_validation(report, out)
    return report
