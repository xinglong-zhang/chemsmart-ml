"""Energy prediction for a new source database with a saved model."""

import csv
import logging
from pathlib import Path

from chemsmart_ml.io.database import SourceDatabase
from chemsmart_ml.models.trained import TrainedModel
from chemsmart_ml.workflows.base import MLWorkflow

logger = logging.getLogger(__name__)


class PredictionWorkflow(MLWorkflow):
    """
    Predict reference-level energies from a low-level database.

    Writes a new CSV of NPZ index, record ID and predicted energy
    (Hartree); an existing output file is never overwritten.

    Attributes:
        filename (Path): Source database.
        model_path (Path): ``model.json`` written by training.
        output (Path): Output CSV file.
    """

    ACTION = "predict"
    BATCH_SIZE = 1024

    def __init__(self, filename, model_path, output):
        super().__init__(output)
        self.filename = Path(filename)
        self.model_path = Path(model_path)

    def run(self) -> dict:
        trained = TrainedModel.from_file(self.model_path)
        features = trained.features
        database = SourceDatabase(
            self.filename,
            include_smiles=features.fingerprint is not None,
            include_positions=features.soap is not None,
        )
        ordered = list(database.records.items())
        records = [record for _, record in ordered]
        descriptor_vectors = features.descriptor_vectors(records)
        with self.output.open("x") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["npz_index", "record_id", "predicted_energy_hartree"]
            )
            for start in range(0, len(ordered), self.BATCH_SIZE):
                stop = start + self.BATCH_SIZE
                batch = records[start:stop]
                x = features.matrix(
                    batch,
                    [vectors[start:stop] for vectors in descriptor_vectors],
                )
                for (key, record), value in zip(
                    ordered[start:stop], trained.correction(x)
                ):
                    writer.writerow(
                        [key, record.record_id, record.energy + value]
                    )
        return dict(samples=len(ordered), output=str(self.output))
