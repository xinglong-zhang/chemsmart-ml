"""Reproducible delta-learning training with group-aware splits."""

import csv
import logging

import numpy as np

from chemsmart_ml.dataset import DataSplit, PairedDataset
from chemsmart_ml.features.builder import FeatureBuilder, FeatureScaler
from chemsmart_ml.features.morgan import MorganFingerprint
from chemsmart_ml.features.soap import SOAPDescriptor
from chemsmart_ml.io.database import ReferenceDatabase, SourceDatabase
from chemsmart_ml.io.manifest import ValidationManifest
from chemsmart_ml.models.delta import DeltaModel
from chemsmart_ml.models.trained import TrainedModel
from chemsmart_ml.workflows.base import MLWorkflow
from chemsmart_ml.workflows.settings import TrainingSettings

logger = logging.getLogger(__name__)


class TrainingWorkflow(MLWorkflow):
    """
    Train a delta model of reference minus low-level energy.

    Writes ``pairing.json``, ``feature-availability.json``, descriptor
    reports, ``model.json``, ``metrics.json`` (MAE in Hartree),
    ``samples.csv`` and, optionally, test-set plots. With cross-
    validation enabled, delegates to ``GroupedCrossValidation`` instead
    of fitting a final model.

    Attributes:
        settings (TrainingSettings): Final training settings.
    """

    ACTION = "train"
    MIN_SAMPLES = 10
    AVAILABILITY_KEYS = ("energy", "homo", "lumo", "dipole")

    def __init__(self, settings: TrainingSettings):
        super().__init__(settings.output)
        self.settings = settings

    @classmethod
    def from_options(cls, **options) -> "TrainingWorkflow":
        return cls(TrainingSettings(**options))

    def run(self) -> dict:
        settings = self.settings
        self._create_output_directory()
        dataset = self._load_dataset()
        self._write_json(
            "feature-availability.json",
            dict(
                samples=len(dataset),
                finite_counts=dataset.finite_counts(self.AVAILABILITY_KEYS),
            ),
        )
        features = self._feature_builder(dataset.elements)
        x = self._feature_matrix(features, dataset.source_records)
        low = dataset.source_energies
        target = dataset.reference_energies
        if not np.isfinite(target).all():
            raise ValueError("Invalid reference energies")
        split = DataSplit.from_groups(
            dataset.group_ids, settings.split_fractions, settings.seed
        )
        split.check_nonempty()
        if settings.cv_folds:
            from chemsmart_ml.workflows.cross_validation import (
                GroupedCrossValidation,
            )

            return GroupedCrossValidation(settings).run(
                x, dataset, split, features
            )
        if settings.model == "kernel-ridge" and settings.kernel_subsample:
            split.subsample_training(
                settings.kernel_max_samples, settings.seed
            )
        trained = self._fit(features, x, target - low, split)
        prediction = low + trained.correction(x)
        trained.write(self.output / "model.json")
        metrics = {
            name: dict(
                n=split.count(name),
                mae_hartree=float(
                    np.abs(
                        prediction[split.mask(name)] - target[split.mask(name)]
                    ).mean()
                ),
            )
            for name in DataSplit.NAMES
        }
        self._write_json("metrics.json", metrics)
        self._write_samples(dataset, split, target, prediction)
        if settings.plot:
            from chemsmart_ml.plotting import PredictionResultsPlotter

            PredictionResultsPlotter(self.output, split="test").plot()
        return metrics

    @property
    def metadata(self) -> dict:
        """Run provenance stored in model and cross-validation reports."""
        return dict(
            split=self.settings.split_fractions,
            seed=self.settings.seed,
            plot=self.settings.plot,
            source=self.settings.source_path,
            reference=self.settings.reference_path,
        )

    def _load_dataset(self) -> PairedDataset:
        """Pair databases, apply any manifest and write ``pairing.json``."""
        settings = self.settings
        source = SourceDatabase(
            settings.filename,
            include_smiles=settings.fingerprint == "morgan",
            include_positions=settings.descriptor == "soap",
        )
        reference = ReferenceDatabase(settings.reference)
        dataset = PairedDataset.from_databases(source, reference)
        if settings.manifest:
            manifest = ValidationManifest.from_file(settings.manifest)
            manifest.verify(source, reference)
            manifest.write(self.output / "validation-manifest.json")
            dataset = dataset.select(manifest)
        self._write_json("pairing.json", dataset.report)
        if dataset.report["rejected"] and not settings.manifest:
            raise ValueError(
                "Pairing rejected composition mismatches; inspect "
                "pairing.json and establish a verified index mapping "
                "before training"
            )
        if len(dataset) < self.MIN_SAMPLES:
            raise ValueError(
                f"Need at least {self.MIN_SAMPLES} paired samples"
            )
        return dataset

    def _feature_builder(self, elements) -> FeatureBuilder:
        settings = self.settings
        fingerprint = None
        if settings.fingerprint == "morgan":
            fingerprint = MorganFingerprint(
                settings.morgan_radius,
                settings.morgan_size,
                settings.morgan_chirality,
                cache=settings.cache,
            )
        soap = None
        if settings.descriptor == "soap":
            soap = SOAPDescriptor(
                elements,
                settings.soap_r_cut,
                settings.soap_n_max,
                settings.soap_l_max,
                settings.soap_sigma,
                settings.soap_aggregation,
                cache=settings.soap_cache,
            )
        return FeatureBuilder(elements, settings.features, fingerprint, soap)

    def _feature_matrix(self, features: FeatureBuilder, records):
        # Validate base features before computing expensive descriptors.
        base = features.base_matrix(records)
        vectors = [
            self._compute_descriptor(descriptor, records)[0]
            for descriptor in features.descriptors
        ]
        return np.column_stack([base, *vectors])

    def _fit(self, features, x, delta, split) -> TrainedModel:
        train_rows = np.flatnonzero(split.mask("train"))
        scaler = FeatureScaler().fit(x[train_rows])
        design = scaler.design_matrix(x)
        delta_model = DeltaModel.from_settings(
            self.settings, base_size=features.base_size + 1
        )
        delta_model.fit(design, delta, train_rows)
        return TrainedModel(features, scaler, delta_model, self.metadata)

    def _write_samples(self, dataset, split, target, prediction):
        with (self.output / "samples.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "npz_index",
                    "split",
                    "source_record",
                    "reference_record",
                    "reference_energy",
                    "prediction",
                ]
            )
            for i, pair in enumerate(dataset):
                writer.writerow(
                    [
                        pair.index,
                        split.labels[i],
                        pair.source.record_id,
                        pair.reference.record_id,
                        target[i],
                        prediction[i],
                    ]
                )
