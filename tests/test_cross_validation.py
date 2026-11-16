import csv
import json
import sqlite3
from collections import defaultdict

import pytest
from click.testing import CliRunner

from chemsmart_ml.cli import build_cli, execute
from test_workflow import database


def test_grouped_cv_preserves_holdout_and_records_folds(tmp_path):
    source = tmp_path / "source.db"
    reference = tmp_path / "reference.db"
    database(source)
    database(reference, True)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE structures SET molecule_id='m0' WHERE structure_id='1'")
    project = tmp_path / "project.yaml"
    project.write_text(f"""data:
  filename: {source}
  reference: {reference}
split:
  train: 0.8
  valid: 0.1
  test: 0.1
train:
  pairing: revqm9-index
  model: hist-gradient-boosting
  max_iter: 2
  cv_folds: 5
  plot: false
  output: cv
""")
    runner = CliRunner()
    cli = build_cli(execute)
    result = runner.invoke(cli, ["-p", str(project), "train"])
    assert result.exit_code == 0, repr(result.exception) + " " + result.output
    report = json.loads((tmp_path / "cv/cross-validation.json").read_text())
    assert len(report["folds"]) == 5 and report["held_out_test_evaluated"] is False
    assert report["model"]["max_iter"] == 2
    with (tmp_path / "cv/cv-assignments.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_group = defaultdict(set)
    for row in rows:
        by_group[row["group_id"]].add((row["partition"], row["validation_fold"]))
    assert all(len(folds) == 1 for folds in by_group.values())
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert {r["validation_fold"] for r in rows if r["partition"] == "development"} == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }
    assert all(not r["validation_fold"] for r in rows if r["partition"] == "holdout")
    assert not (tmp_path / "cv/model.json").exists()
    assert not (tmp_path / "cv/metrics.json").exists()
    assert all("train_mae_kcal_mol" in fold for fold in report["folds"])
    result = runner.invoke(
        cli,
        [
            "-p",
            str(project),
            "train",
            "--cv-folds",
            "0",
            "-o",
            str(tmp_path / "normal"),
        ],
    )
    assert result.exit_code == 0, repr(result.exception) + " " + result.output
    original = {
        r["npz_index"]: r["split"]
        for r in csv.DictReader((tmp_path / "normal/samples.csv").open())
    }
    assert {r["npz_index"] for r in rows if r["partition"] == "holdout"} == {
        i for i, split in original.items() if split == "test"
    }


def test_cross_validation_rejects_unsupported_model(tmp_path):
    from chemsmart_ml.workflow import train

    with pytest.raises(ValueError, match="Cross-validation currently supports"):
        train("unused", "unused", tmp_path / "cv", model="ridge", cv_folds=5)
