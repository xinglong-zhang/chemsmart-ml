import json
import sqlite3

import pytest
from click.testing import CliRunner

from chemsmart_ml.cli import build_cli, execute
from chemsmart_ml.workflows import TrainingSettings
from tests.helpers import (
    assert_predictions_match_training,
    predict,
    read_csv,
    train,
)


def test_settings_reject_cv_for_unsupported_model(tmp_path):
    with pytest.raises(
        ValueError, match="Cross-validation currently supports"
    ):
        TrainingSettings(
            "unused", "unused", tmp_path, model="ridge", cv_folds=5
        )


def test_settings_learning_rate_defaults_by_model():
    assert TrainingSettings("a", "b", "c", model="mlp").learning_rate == 0.001
    assert TrainingSettings("a", "b", "c").learning_rate == 0.1


def test_linear_train_predict(paired_databases, tmp_path):
    source, reference = paired_databases
    metrics = train(source, reference, tmp_path / "run")
    assert metrics["test"]["mae_hartree"] < 1e-10
    predict(source, tmp_path / "run/model.json", tmp_path / "pred.csv")
    assert len(read_csv(tmp_path / "pred.csv")) == 120
    with pytest.raises(FileExistsError):
        train(source, reference, tmp_path / "run")


def test_ridge_roundtrip(paired_databases, tmp_path):
    source, reference = paired_databases
    train(source, reference, tmp_path / "ridge", model="ridge", alpha=0.1)
    model = json.loads((tmp_path / "ridge/model.json").read_text())
    assert model["model"] == "ridge-delta"
    assert_predictions_match_training(
        source, tmp_path / "ridge", tmp_path / "ridge.csv"
    )


def test_orbitals_without_dipole(paired_databases, tmp_path):
    source, reference = paired_databases
    with sqlite3.connect(source) as db:
        db.execute("UPDATE record_structures SET dipole_moment_magnitude=NULL")
    train(source, reference, tmp_path / "orbitals", features="orbitals")
    predict(
        source, tmp_path / "orbitals/model.json", tmp_path / "orbitals.csv"
    )
    counts = json.loads(
        (tmp_path / "orbitals/feature-availability.json").read_text()
    )
    assert counts["finite_counts"]["dipole"] == 0
    model = json.loads((tmp_path / "orbitals/model.json").read_text())
    assert model["electronic"] == "orbitals"


@pytest.mark.parametrize("name", ["mlp", "kernel-ridge"])
def test_nonlinear_training_and_portable_prediction(
    paired_databases, tmp_path, name
):
    pytest.importorskip("sklearn")
    source, reference = paired_databases
    options = [
        "-f",
        str(source),
        "-r",
        str(reference),
        "train",
        "--pairing",
        "revqm9-index",
        "--model",
        name,
        "--max-iter",
        "10",
        "--hidden-size",
        "8",
        "--no-plot",
        "-o",
        str(tmp_path / "trained"),
    ]
    if name == "kernel-ridge":
        options += ["--kernel-subsample", "--kernel-max-samples", "50"]
    result = CliRunner().invoke(build_cli(execute), options)
    assert result.exit_code == 0, repr(result.exception) + " " + result.output
    model = json.loads((tmp_path / "trained/model.json").read_text())
    assert model["nonlinear"]["kind"] == name
    if name == "kernel-ridge":
        assert len(model["nonlinear"]["support_vectors"]) == 50
        labels = [
            r["split"] for r in read_csv(tmp_path / "trained/samples.csv")
        ]
        assert labels.count("train") == 50 and "unused" in labels
    assert_predictions_match_training(
        source, tmp_path / "trained", tmp_path / "prediction.csv"
    )


def test_morgan_train_predict_and_prepare(
    smiles_databases, tmp_path, config_dir
):
    pytest.importorskip("rdkit")
    source, reference = smiles_databases
    project = tmp_path / "ml.yaml"
    project.write_text(
        f"data:\n  filename: {source}\n  reference: {reference}\n"
        "train:\n  pairing: revqm9-index\n  fingerprint: morgan\n"
        "  morgan_size: 32\n  output: train\n"
        "prepare:\n  morgan_size: 32\n  output: prepared\n"
    )
    cli = build_cli(execute)
    result = CliRunner().invoke(cli, ["-p", str(project), "prepare"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "prepared/fingerprints.json").read_text())
    assert report["generated"] == 1
    result = CliRunner().invoke(
        cli, ["-p", str(project), "train", "--model", "ridge"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "train/fingerprints.json").read_text())
    assert report["generated"] == 0
    model = json.loads((tmp_path / "train/model.json").read_text())
    assert model["fingerprint"]["size"] == 32
    assert len(model["mean"]) == 35
    predict(source, tmp_path / "train/model.json", tmp_path / "pred.csv")
    assert len(read_csv(tmp_path / "pred.csv")) == 120


def test_soap_tree_train_predict_and_split_yaml(
    geometry_databases, tmp_path, config_dir
):
    pytest.importorskip("chemsmart.analysis.soap")
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")
    source, reference = geometry_databases
    project = tmp_path / "project.yaml"
    project.write_text(
        f"data:\n  filename: {source}\n  reference: {reference}\n"
        "split:\n  train: 0.6\n  valid: 0.2\n  test: 0.2\n"
        "train:\n  pairing: revqm9-index\n  descriptor: soap\n"
        "  fingerprint: morgan\n  morgan_size: 32\n"
        "  model: hist-gradient-boosting\n  max_iter: 4\n  plot: false\n"
        "  output: result\n"
    )
    cli = build_cli(execute)
    result = CliRunner().invoke(
        cli,
        [
            "-p",
            str(project),
            "train",
            "--train-fraction",
            "0.7",
            "--valid-fraction",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output
    model = json.loads((tmp_path / "result/model.json").read_text())
    assert model["split"] == {"train": 0.7, "valid": 0.1, "test": 0.2}
    assert model["soap"]["species"] == ["C", "H"]
    assert model["model"] == "hist-gradient-boosting-delta"
    assert model["fingerprint"]["size"] == 32 and len(model["mean"]) > 35
    assert_predictions_match_training(
        source, tmp_path / "result", tmp_path / "pred.csv"
    )
    result = CliRunner().invoke(
        cli,
        [
            "-p",
            str(project),
            "train",
            "--test-fraction",
            "0.3",
            "-o",
            str(tmp_path / "bad"),
        ],
    )
    assert result.exit_code != 0 and "sum to 1" in result.output


def test_ridge_with_soap_roundtrip(geometry_databases, tmp_path, config_dir):
    pytest.importorskip("chemsmart.analysis.soap")
    source, reference = geometry_databases
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
    assert_predictions_match_training(
        source, tmp_path / "run", tmp_path / "predict.csv"
    )


def test_train_automatic_plot_and_opt_out(paired_databases, tmp_path):
    pytest.importorskip("matplotlib")
    source, reference = paired_databases
    train(source, reference, tmp_path / "auto")
    assert (tmp_path / "auto/plots/test-results.png").is_file()
    assert (tmp_path / "auto/plots/test-results.pdf").is_file()
    stats = json.loads(
        (tmp_path / "auto/plots/test-statistics.json").read_text()
    )
    metrics = json.loads((tmp_path / "auto/metrics.json").read_text())
    assert stats["n"] == metrics["test"]["n"]
    project = tmp_path / "project.yaml"
    project.write_text(
        f"data:\n  filename: {source}\n  reference: {reference}\n"
        "train:\n  pairing: revqm9-index\n  plot: false\n  output: disabled\n"
    )
    cli = build_cli(execute)
    result = CliRunner().invoke(cli, ["-p", str(project), "train"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "disabled/plots").exists()
    result = CliRunner().invoke(
        cli,
        ["-p", str(project), "train", "--plot", "-o", str(tmp_path / "over")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "over/plots/test-results.png").exists()
    result = CliRunner().invoke(
        cli,
        [
            "-f",
            str(source),
            "-r",
            str(reference),
            "train",
            "--pairing",
            "revqm9-index",
            "--no-plot",
            "-o",
            str(tmp_path / "no-plot"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "no-plot/plots").exists()
