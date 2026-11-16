import csv
import json
import sqlite3

import numpy as np
import pytest
from click.testing import CliRunner

from chemsmart_ml.cli import build_cli, execute
from chemsmart_ml.workflow import train, predict
from chemsmart_ml.soap_features import soap_config, cached_soap
from chemsmart_ml.tree_model import fit_tree, predict_tree
from test_workflow import database
from test_fingerprints import add_smiles


def source_with_geometry(path):
    database(path)
    add_smiles(path)
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE structures ADD COLUMN positions_json TEXT")
        db.execute(
            "UPDATE structures SET positions_json=?",
            (json.dumps([[0, 0, 0], [1.1, 0, 0]]),),
        )


def test_soap_cache_and_geometry_invariance(tmp_path):
    row = dict(record_id="test", atoms=["C", "H"], positions=[[0, 0, 0], [1.1, 0, 0]])
    cfg = soap_config(["C", "H"])
    first, report = cached_soap([row], cfg, tmp_path / "cache.db")
    assert report["generated"] == 1 and first.shape[1] > 0
    again, report = cached_soap([row], cfg, tmp_path / "cache.db")
    assert report["cache_hits"] == 1
    np.testing.assert_allclose(first, again)
    translated = dict(row, positions=[[5, 2, 1], [6.1, 2, 1]])
    new, _ = cached_soap([translated], cfg, tmp_path / "cache.db")
    np.testing.assert_allclose(first, new, atol=1e-10)
    bad = dict(row, positions=[[0, 0, 0]])
    with pytest.raises(ValueError, match="Invalid geometry"):
        cached_soap([bad], cfg, tmp_path / "cache.db")


def test_tree_json_matches_sklearn_and_non_linearity():
    x = np.linspace(-2, 2, 300)[:, None]
    y = x[:, 0] ** 2
    artifact = fit_tree(x, y, 42, 60, 0.1, 15)
    out = predict_tree(x, json.loads(json.dumps(artifact)))
    assert np.mean(abs(out - y)) < 0.15


def test_soap_tree_train_predict_and_split_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMSMART_CONFIG_DIR", str(tmp_path / "config"))
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    source_with_geometry(a)
    database(b, True)
    cfg = tmp_path / "project.yaml"
    cfg.write_text(
        f"data:\n  filename: {a}\n  reference: {b}\nsplit:\n  train: 0.6\n  valid: 0.2\n  test: 0.2\ntrain:\n  pairing: revqm9-index\n  descriptor: soap\n  fingerprint: morgan\n  morgan_size: 32\n  model: hist-gradient-boosting\n  max_iter: 4\n  plot: false\n  output: result\n"
    )
    result = CliRunner().invoke(
        build_cli(execute),
        ["-p", str(cfg), "train", "--train-fraction", "0.7", "--valid-fraction", "0.1"],
    )
    assert result.exit_code == 0, result.output
    model = json.loads((tmp_path / "result/model.json").read_text())
    assert model["split"] == {"train": 0.7, "valid": 0.1, "test": 0.2}
    assert (
        model["soap"]["species"] == ["C", "H"]
        and model["model"] == "hist-gradient-boosting-delta"
    )
    assert model["fingerprint"]["size"] == 32 and len(model["mean"]) > 35
    output = tmp_path / "pred.csv"
    predict(a, tmp_path / "result/model.json", output)
    predictions = {
        r["record_id"]: float(r["predicted_energy_hartree"])
        for r in csv.DictReader(output.open())
    }
    for row in csv.DictReader((tmp_path / "result/samples.csv").open()):
        assert predictions[row["source_record"]] == pytest.approx(
            float(row["prediction"]), abs=1e-9
        )
    result = CliRunner().invoke(
        build_cli(execute),
        [
            "-p",
            str(cfg),
            "train",
            "--test-fraction",
            "0.3",
            "-o",
            str(tmp_path / "bad"),
        ],
    )
    assert result.exit_code != 0 and "sum to 1" in result.output


def test_ridge_with_soap_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMSMART_CONFIG_DIR", str(tmp_path / "config"))
    source, reference = tmp_path / "source.db", tmp_path / "reference.db"
    source_with_geometry(source)
    database(reference, True)
    train(
        source,
        reference,
        tmp_path / "run",
        descriptor="soap",
        model="ridge",
        alpha=1e-6,
        plot=False,
    )
    model = json.loads((tmp_path / "run/model.json").read_text())
    assert model["model"] == "ridge-delta"
    assert model["soap"]["method"] == "soap"
    predict(source, tmp_path / "run/model.json", tmp_path / "predict.csv")
    predicted = {
        r["record_id"]: float(r["predicted_energy_hartree"])
        for r in csv.DictReader((tmp_path / "predict.csv").open())
    }
    for row in csv.DictReader((tmp_path / "run/samples.csv").open()):
        assert predicted[row["source_record"]] == pytest.approx(
            float(row["prediction"]), abs=1e-9
        )
