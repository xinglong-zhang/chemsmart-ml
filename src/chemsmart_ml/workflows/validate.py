"""Task-specific input screening producing a reusable sample manifest."""

import csv
import logging
from pathlib import Path

from chemsmart_ml.dataset import PairedDataset, SamplePair
from chemsmart_ml.features.builder import FeatureBuilder
from chemsmart_ml.features.morgan import MorganFingerprint
from chemsmart_ml.io.database import ReferenceDatabase, SourceDatabase
from chemsmart_ml.io.manifest import ValidationManifest
from chemsmart_ml.workflows.base import MLWorkflow

logger = logging.getLogger(__name__)


class DatasetValidation(MLWorkflow):
    """
    Screen paired samples for the inputs a training run will need.

    Checks pairing, required numeric source features, the reference
    energy and, for Morgan fingerprints, RDKit SMILES parsing. This is
    not a geometry or convergence audit. Writes ``valid-samples.json``
    (a manifest for ``train``/``prepare``), ``excluded.csv`` and
    ``validation.json``.

    Attributes:
        filename (Path): Source database.
        reference (Path): Reference database.
        features (str): Feature set whose inputs are required.
        fingerprint (str): 'none' or 'morgan'.
        output (Path): New output directory.
    """

    ACTION = "validate"
    FINGERPRINTS = ("none", "morgan")
    EXCLUDED_COLUMNS = [
        "npz_index",
        "idx",
        "source_record",
        "reference_record",
        "reasons",
        "smiles",
    ]
    SCOPE = (
        "Pairing, required numeric features and RDKit SMILES parsing; "
        "not a geometry/convergence audit"
    )

    def __init__(
        self,
        filename,
        reference,
        output,
        features: str = "orbitals",
        fingerprint: str = "morgan",
    ):
        super().__init__(output)
        if fingerprint not in self.FINGERPRINTS:
            raise ValueError("Unsupported fingerprint")
        if features not in FeatureBuilder.ELECTRONIC_KEYS:
            raise ValueError(f"Unsupported feature set {features!r}")
        self.filename = Path(filename)
        self.reference = Path(reference)
        self.features = features
        self.fingerprint = fingerprint

    @property
    def required_source_keys(self):
        return ("energy",) + FeatureBuilder.ELECTRONIC_KEYS[self.features]

    def run(self) -> dict:
        self._create_output_directory()
        source = SourceDatabase(
            self.filename, include_smiles=self.fingerprint == "morgan"
        )
        reference = ReferenceDatabase(self.reference)
        dataset = PairedDataset.from_databases(source, reference)
        accepted, excluded = [], []
        smiles_valid = {}
        for pair in dataset:
            reasons = self._rejection_reasons(pair, smiles_valid)
            item = dict(
                npz_index=pair.index,
                idx=pair.index + 1,
                source_record=pair.source.record_id,
                reference_record=pair.reference.record_id,
            )
            if reasons:
                excluded.append(
                    dict(
                        **item,
                        reasons="; ".join(reasons),
                        smiles=pair.source.smiles or "",
                    )
                )
            else:
                accepted.append(item)
        for row in dataset.report["rejected"]:
            excluded.append(
                dict(
                    npz_index=row["index"],
                    idx=row["index"] + 1,
                    source_record="",
                    reference_record="",
                    reasons=row["reason"],
                    smiles="",
                )
            )
        manifest = ValidationManifest(
            source=source.identity,
            reference=reference.identity,
            electronic=FeatureBuilder.ELECTRONIC_ARTIFACT_VALUES[
                self.features
            ],
            fingerprint=self.fingerprint,
            samples=accepted,
        )
        manifest_path = self.output / "valid-samples.json"
        manifest.write(manifest_path)
        with (self.output / "excluded.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.EXCLUDED_COLUMNS)
            writer.writeheader()
            writer.writerows(excluded)
        report = dict(
            valid=len(accepted),
            excluded=len(excluded),
            pairing=dataset.report,
            scope=self.SCOPE,
            manifest=str(manifest_path.resolve()),
        )
        self._write_json("validation.json", report)
        return report

    def _rejection_reasons(self, pair: SamplePair, smiles_valid: dict):
        reasons = [
            f"missing/nonfinite source {key}"
            for key in self.required_source_keys
            if not pair.source.has_finite(key)
        ]
        if not pair.reference.has_finite("energy"):
            reasons.append("missing/nonfinite reference energy")
        if self.fingerprint == "morgan":
            smiles = pair.source.smiles
            if smiles not in smiles_valid:
                smiles_valid[smiles] = MorganFingerprint.is_valid_smiles(
                    smiles
                )
            if not smiles_valid[smiles]:
                reasons.append("missing/invalid RDKit SMILES")
        return reasons
