"""Grouped development-set cross-validation with an untouched holdout."""

import csv
import json
import logging
from pathlib import Path

import numpy as np

from chemsmart_ml.constants import HARTREE_TO_KCAL_MOL
from chemsmart_ml.dataset import DataSplit, PairedDataset
from chemsmart_ml.features.builder import FeatureBuilder, FeatureScaler
from chemsmart_ml.models.delta import DeltaModel
from chemsmart_ml.workflows.settings import TrainingSettings

logger = logging.getLogger(__name__)


class GroupedCrossValidation:
    """
    K-fold cross-validation over molecular groups of train+valid samples.

    The test partition is held out and never evaluated. Each fold
    standardises features and fits the delta model on its own training
    groups only. No final model is written.

    Attributes:
        settings (TrainingSettings): Training settings; ``cv_folds`` sets
            the number of folds.
        output (Path): Directory for the report and fold assignments.
    """

    REPORT_FILENAME = "cross-validation.json"
    ASSIGNMENTS_FILENAME = "cv-assignments.csv"

    def __init__(self, settings: TrainingSettings, output=None):
        self.settings = settings
        self.output = Path(output if output is not None else settings.output)

    @property
    def n_splits(self) -> int:
        return self.settings.cv_folds

    def run(
        self,
        x: np.ndarray,
        dataset: PairedDataset,
        split: DataSplit,
        features: FeatureBuilder,
    ) -> dict:
        """
        Fit and score each fold, then write the report and assignments.

        Args:
            x: Feature matrix of all samples.
            dataset: Paired samples with group IDs.
            split: Partition labels; 'test' samples are held out.
            features: Feature definition used to build ``x``.
        """
        try:
            from sklearn.model_selection import GroupKFold
        except ImportError as exc:
            raise ValueError(
                "Cross-validation requires scikit-learn>=1.6"
            ) from exc
        if self.n_splits < 2:
            raise ValueError("Cross-validation requires at least two folds")
        development = np.flatnonzero(~split.mask("test"))
        holdout = np.flatnonzero(split.mask("test"))
        group_ids = np.asarray(dataset.group_ids)
        if (
            not len(holdout)
            or len(np.unique(group_ids[development])) < self.n_splits
        ):
            raise ValueError("Not enough groups or no held-out test samples")
        low = dataset.source_energies
        target = dataset.reference_energies
        splitter = GroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.settings.seed,
        )
        assignment = np.full(len(split), -1, dtype=int)
        folds = []
        for fold, (local_train, local_valid) in enumerate(
            splitter.split(development, groups=group_ids[development]),
            start=1,
        ):
            train = development[local_train]
            valid = development[local_valid]
            assignment[valid] = fold
            logger.info(
                f"Cross-validation fold {fold}/{self.n_splits}: "
                f"{len(train)} training, {len(valid)} validation samples"
            )
            folds.append(
                self._score_fold(fold, x, low, target, train, valid, features)
            )
        if np.any(assignment[development] < 1) or np.any(
            assignment[holdout] != -1
        ):
            raise ValueError("Invalid fold assignment")
        self._write_assignments(dataset, assignment)
        report = self._report(features, folds, development, holdout)
        (self.output / self.REPORT_FILENAME).write_text(
            json.dumps(report, indent=2)
        )
        if self.settings.plot:
            from chemsmart_ml.plotting import CrossValidationPlotter

            CrossValidationPlotter(report).plot(self.output)
        return report

    def _score_fold(self, fold, x, low, target, train, valid, features):
        scaler = FeatureScaler().fit(x[train])
        train_design = scaler.design_matrix(x[train])
        valid_design = scaler.design_matrix(x[valid])
        model = DeltaModel.from_settings(
            self.settings, base_size=features.base_size + 1
        )
        model.fit(
            train_design, target[train] - low[train], np.arange(len(train))
        )
        training_error = (
            low[train] + model.predict(train_design) - target[train]
        )
        error = low[valid] + model.predict(valid_design) - target[valid]
        mae = float(np.abs(error).mean())
        return dict(
            fold=fold,
            train_n=len(train),
            validation_n=len(valid),
            train_mae_kcal_mol=float(
                np.abs(training_error).mean() * HARTREE_TO_KCAL_MOL
            ),
            mae_hartree=mae,
            mae_kcal_mol=float(mae * HARTREE_TO_KCAL_MOL),
            rmse_kcal_mol=float(
                np.sqrt(np.mean(error**2)) * HARTREE_TO_KCAL_MOL
            ),
        )

    def _write_assignments(self, dataset, assignment):
        path = self.output / self.ASSIGNMENTS_FILENAME
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
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
            for i, pair in enumerate(dataset):
                held_out = assignment[i] == -1
                writer.writerow(
                    [
                        pair.index,
                        pair.index + 1,
                        pair.source.record_id,
                        pair.reference.record_id,
                        dataset.group_ids[i],
                        "holdout" if held_out else "development",
                        "" if held_out else int(assignment[i]),
                    ]
                )

    def _report(self, features, folds, development, holdout) -> dict:
        settings = self.settings
        validation = np.array([f["mae_kcal_mol"] for f in folds])
        training = np.array([f["train_mae_kcal_mol"] for f in folds])
        fingerprint, soap = features.fingerprint, features.soap
        model_settings = dict(
            model=settings.model,
            max_iter=settings.max_iter,
            learning_rate=settings.learning_rate,
            max_leaf_nodes=settings.max_leaf_nodes,
            features=features.electronic,
            fingerprint=fingerprint.settings if fingerprint else None,
            soap=soap.settings if soap else None,
            split=settings.split_fractions,
            source=settings.source_path,
            reference=settings.reference_path,
        )
        return dict(
            n_splits=self.n_splits,
            seed=settings.seed,
            development_samples=len(development),
            held_out_test_samples=len(holdout),
            held_out_test_evaluated=False,
            model=model_settings,
            folds=folds,
            train_mae_mean_kcal_mol=float(training.mean()),
            validation_mae_mean_kcal_mol=float(validation.mean()),
            validation_mae_std_kcal_mol=float(validation.std(ddof=1)),
        )
