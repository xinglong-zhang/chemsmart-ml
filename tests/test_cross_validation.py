import json
import sqlite3
from collections import defaultdict

import pytest
from click.testing import CliRunner

from chemsmart_ml.cli import build_cli, execute
from tests.helpers import read_csv, train

pytest.importorskip("sklearn")


def test_grouped_cv_preserves_holdout_and_records_folds(
    paired_databases, tmp_path
):
    source, reference = paired_databases
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE structures SET molecule_id='m0' WHERE structure_id='1'"
        )
    project = tmp_path / "project.yaml"
    project.write_text(
        f"data:\n  filename: {source}\n  reference: {reference}\n"
        "split:\n  train: 0.8\n  valid: 0.1\n  test: 0.1\n"
        "train:\n  pairing: revqm9-index\n  model: hist-gradient-boosting\n"
        "  max_iter: 2\n  cv_folds: 5\n  plot: false\n  output: cv\n"
    )
    runner = CliRunner()
    cli = build_cli(execute)
    result = runner.invoke(cli, ["-p", str(project), "train"])
    assert result.exit_code == 0, repr(result.exception) + " " + result.output
    report = json.loads((tmp_path / "cv/cross-validation.json").read_text())
    assert len(report["folds"]) == 5
    assert report["held_out_test_evaluated"] is False
    assert report["model"]["max_iter"] == 2
    rows = read_csv(tmp_path / "cv/cv-assignments.csv")
    by_group = defaultdict(set)
    for row in rows:
        by_group[row["group_id"]].add(
            (row["partition"], row["validation_fold"])
        )
    assert all(len(folds) == 1 for folds in by_group.values())
    assert rows[0]["group_id"] == rows[1]["group_id"]
    development = [r for r in rows if r["partition"] == "development"]
    holdout = [r for r in rows if r["partition"] == "holdout"]
    assert {r["validation_fold"] for r in development} == set("12345")
    assert all(not r["validation_fold"] for r in holdout)
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
        for r in read_csv(tmp_path / "normal/samples.csv")
    }
    assert {r["npz_index"] for r in holdout} == {
        i for i, split in original.items() if split == "test"
    }


def test_plot_saved_cv_report_without_training(tmp_path):
    pytest.importorskip("matplotlib")
    report = {
        "folds": [
            {
                "fold": i,
                "train_mae_kcal_mol": 1.2 + i * 0.01,
                "mae_kcal_mol": 1.5 + i * 0.01,
            }
            for i in range(1, 6)
        ]
    }
    (tmp_path / "cross-validation.json").write_text(json.dumps(report))
    cli = build_cli(execute)
    result = CliRunner().invoke(cli, ["plot", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "plots/cross-validation.png").is_file()
    assert (tmp_path / "plots/cross-validation.pdf").is_file()
    assert "validation_mae_mean_kcal_mol" in result.output
    result = CliRunner().invoke(cli, ["plot", "--workdir", str(tmp_path)])
    assert result.exit_code != 0 and "already exist" in result.output


def test_cross_validation_automatically_writes_plot(
    paired_databases, tmp_path
):
    pytest.importorskip("matplotlib")
    source, reference = paired_databases
    train(
        source,
        reference,
        tmp_path / "cv",
        model="hist-gradient-boosting",
        max_iter=2,
        cv_folds=5,
        plot=True,
    )
    assert (tmp_path / "cv/cross-validation.png").is_file()
    assert (tmp_path / "cv/cross-validation.pdf").is_file()
