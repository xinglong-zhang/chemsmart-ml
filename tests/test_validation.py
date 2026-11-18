import sqlite3

import pytest

from chemsmart_ml.io import (
    ReferenceDatabase,
    SourceDatabase,
    ValidationManifest,
)
from chemsmart_ml.workflows import DatasetValidation, FingerprintPreparation
from tests.helpers import read_csv, train


def test_full_screen_manifest_prepare_train(
    smiles_databases, tmp_path, config_dir
):
    pytest.importorskip("rdkit")
    source, reference = smiles_databases
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE molecules SET smiles='invalid!' WHERE molecule_id='m3'"
        )
        db.execute("UPDATE records SET homo_energy=NULL WHERE record_id='4'")
    report = DatasetValidation(
        source, reference, tmp_path / "validation"
    ).run()
    assert report["valid"] == 118 and report["excluded"] == 2
    manifest = tmp_path / "validation/valid-samples.json"
    excluded = read_csv(tmp_path / "validation/excluded.csv")
    assert {int(row["idx"]) for row in excluded} == {4, 5}
    FingerprintPreparation.from_options(
        source, tmp_path / "prepared", morgan_size=32, manifest=manifest
    ).run()
    train(
        source,
        reference,
        tmp_path / "trained",
        features="orbitals",
        fingerprint="morgan",
        morgan_size=32,
        model="ridge",
        manifest=manifest,
    )
    rows = read_csv(tmp_path / "trained/samples.csv")
    assert len(rows) == 118 and not {"3", "4"} & {r["npz_index"] for r in rows}
    # The same filtered sample set gives a fair numerical baseline.
    train(
        source,
        reference,
        tmp_path / "baseline",
        features="orbitals",
        manifest=manifest,
    )
    base = read_csv(tmp_path / "baseline/samples.csv")
    assert [(r["npz_index"], r["split"]) for r in base] == [
        (r["npz_index"], r["split"]) for r in rows
    ]
    with sqlite3.connect(source) as db:
        db.execute("UPDATE records SET homo_energy=0 WHERE record_id='4'")
    with pytest.raises(ValueError, match="source database"):
        ValidationManifest.from_file(manifest).verify(
            SourceDatabase(source), ReferenceDatabase(reference)
        )
