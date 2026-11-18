"""Plain test helpers: builder functions that need no pytest fixtures.

Fixtures live in ``tests/fixtures/`` (loaded by ``tests/conftest.py``).
"""

import csv
import json
import sqlite3

import pytest

from chemsmart_ml.cli import build_cli
from chemsmart_ml.io import MolecularRecord
from chemsmart_ml.workflows import PredictionWorkflow, TrainingWorkflow

N_SAMPLES = 120


def create_database(path, reference=False):
    """Write a minimal CHEMSMART database of ``N_SAMPLES`` C/H records.

    Reference energies are an exact linear function of source energies.
    """
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE records(record_id TEXT, source TEXT, total_energy REAL,
            homo_energy REAL, lumo_energy REAL);
        CREATE TABLE structures(structure_id TEXT, molecule_id TEXT,
            chemical_symbols_json TEXT);
        CREATE TABLE record_structures(record_id TEXT, structure_id TEXT,
            index_in_record INTEGER, structure_index_in_file INTEGER,
            dipole_moment_magnitude REAL);
        """)
    for i in range(N_SAMPLES):
        energy = -10 - i * 0.1
        if reference:
            energy = energy * 1.2 + 3
        db.execute(
            "INSERT INTO records VALUES(?,?,?,?,?)",
            (str(i), f"/opt_idx{i + 1}", energy, -1, 1),
        )
        db.execute(
            "INSERT INTO structures VALUES(?,?,?)",
            (str(i), f"m{i}", json.dumps(["C", "H"])),
        )
        db.execute(
            "INSERT INTO record_structures VALUES(?,?,?,?,?)",
            (str(i), str(i), 1, i + 1, 0.5),
        )
    db.commit()
    db.close()


def add_smiles(path, smiles="C"):
    """Add a molecules table giving every molecule the same SMILES."""
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE molecules(molecule_id TEXT PRIMARY KEY, smiles TEXT)"
        )
        db.executemany(
            "INSERT INTO molecules VALUES (?,?)",
            [(f"m{i}", smiles) for i in range(N_SAMPLES)],
        )


def add_geometry(path):
    """Add identical C-H coordinates to every structure."""
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE structures ADD COLUMN positions_json TEXT")
        db.execute(
            "UPDATE structures SET positions_json=?",
            (json.dumps([[0, 0, 0], [1.1, 0, 0]]),),
        )


def train(filename, reference, output, **options):
    """Run a training workflow with keyword overrides of the settings."""
    return TrainingWorkflow.from_options(
        filename=filename, reference=reference, output=output, **options
    ).run()


def predict(filename, model_path, output):
    return PredictionWorkflow(filename, model_path, output).run()


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def predictions_by_record(path):
    """Predicted energies keyed by source record ID."""
    return {
        row["record_id"]: float(row["predicted_energy_hartree"])
        for row in read_csv(path)
    }


def assert_predictions_match_training(source, run, output):
    """Predicting the source database reproduces training predictions."""
    predict(source, run / "model.json", output)
    predicted = predictions_by_record(output)
    for row in read_csv(run / "samples.csv"):
        assert predicted[row["source_record"]] == pytest.approx(
            float(row["prediction"]), abs=1e-9
        )


def make_record(
    record_id="m",
    atoms=("C",),
    energy=-1.0,
    homo=None,
    lumo=None,
    dipole=None,
    molecule_id=None,
    positions=None,
    smiles=None,
):
    """Build a record; the molecule ID defaults to the record ID."""
    return MolecularRecord(
        record_id=record_id,
        energy=energy,
        homo=homo,
        lumo=lumo,
        dipole=dipole,
        molecule_id=record_id if molecule_id is None else molecule_id,
        atoms=list(atoms),
        positions=positions,
        smiles=smiles,
    )


def recording_cli(calls):
    """CLI whose dispatch records (action, options) instead of running."""
    return build_cli(lambda action, options: calls.append((action, options)))
