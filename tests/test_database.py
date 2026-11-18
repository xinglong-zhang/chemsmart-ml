import sqlite3

import numpy as np
import pytest

from chemsmart_ml.dataset import DataSplit, PairedDataset, SamplePair
from chemsmart_ml.io import ReferenceDatabase, SourceDatabase
from tests.helpers import create_database, make_record


def test_source_and_reference_indices_pair(paired_databases):
    source = SourceDatabase(paired_databases[0])
    reference = ReferenceDatabase(paired_databases[1])
    assert set(source.records) == set(range(120))
    assert set(reference.records) == set(range(120))
    dataset = PairedDataset.from_databases(source, reference)
    assert dataset.report["matched"] == 120 and len(dataset) == 120
    assert dataset.elements == ["C", "H"]


def test_optional_columns_are_read(geometry_databases):
    plain = SourceDatabase(geometry_databases[0]).records[0]
    assert plain.smiles is None and plain.positions is None
    full = SourceDatabase(
        geometry_databases[0], include_smiles=True, include_positions=True
    ).records[0]
    assert full.smiles == "C"
    assert full.positions == [[0, 0, 0], [1.1, 0, 0]]


def test_group_transitivity():
    pairs = [
        SamplePair(0, make_record("A"), make_record("B")),
        SamplePair(1, make_record("C"), make_record("B")),
    ]
    assert len(set(PairedDataset(pairs, {}).group_ids)) == 1


def test_duplicate_and_composition(tmp_path):
    path = tmp_path / "a.db"
    create_database(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE records SET source='/opt_idx1' WHERE record_id='1'")
    with pytest.raises(ValueError, match="Duplicate"):
        SourceDatabase(path).records
    dataset = PairedDataset.from_records(
        {0: make_record(atoms=["C"])}, {0: make_record(atoms=["N"])}
    )
    assert not len(dataset) and len(dataset.report["rejected"]) == 1


def test_split_keeps_groups_together_and_subsamples():
    groups = [f"g{i // 3}" for i in range(300)]
    fractions = dict(train=0.8, valid=0.1, test=0.1)
    split = DataSplit.from_groups(groups, fractions, seed=42)
    for start in range(0, 300, 3):
        assert len(set(split.labels[start : start + 3])) == 1
    np.testing.assert_array_equal(
        split.labels, DataSplit.from_groups(groups, fractions, 42).labels
    )
    n_train = split.count("train")
    split.subsample_training(10, seed=1)
    assert split.count("train") == 10
    assert split.count("unused") == n_train - 10
