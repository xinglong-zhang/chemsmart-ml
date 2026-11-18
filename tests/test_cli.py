import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from chemsmart_ml.cli import build_cli
from chemsmart_ml.cli.options import complete_project
from tests.helpers import recording_cli


def test_yaml_precedence_and_paths(paired_databases, tmp_path):
    source, _ = paired_databases
    project = tmp_path / "ml.yaml"
    project.write_text(
        "data:\n  filename: source.db\n  reference: reference.db\n"
        "train:\n  pairing: revqm9-index\n  model: ridge\n  alpha: 2.0\n"
        "  output: run\n"
    )
    calls = []
    cli = recording_cli(calls)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-p", str(project), "train", "--model", "linear", "--alpha", "0.5"],
    )
    assert result.exit_code == 0, result.output
    action, options = calls[-1]
    assert action == "train"
    assert options["model"] == "linear" and options["alpha"] == 0.5
    assert options["output"] == str(tmp_path / "run")
    assert options["filename"] == str(source)
    result = runner.invoke(cli, ["-p", str(project), "train"])
    assert result.exit_code == 0, result.output
    assert calls[-1][1]["model"] == "ridge"
    project.write_text("train:\n  typo: 3\n")
    result = runner.invoke(cli, ["-p", str(project), "train"])
    assert result.exit_code != 0 and "Unknown keys" in result.output


def test_options_match_training_settings(paired_databases):
    from dataclasses import fields

    from chemsmart_ml.workflows import TrainingSettings

    source, reference = paired_databases
    calls = []
    result = CliRunner().invoke(
        recording_cli(calls),
        [
            "-f",
            str(source),
            "-r",
            str(reference),
            "train",
            "--pairing",
            "revqm9-index",
        ],
    )
    assert result.exit_code == 0, result.output
    assert set(calls[-1][1]) == {f.name for f in fields(TrainingSettings)}


def test_plot_cli_without_database(tmp_path):
    pytest.importorskip("matplotlib")
    (tmp_path / "samples.csv").write_text(
        "split,reference_energy,prediction\ntest,-10,-9.99\ntrain,-20,-20\n"
    )
    cli = build_cli(lambda *args: pytest.fail("Plot must not create a job"))
    result = CliRunner().invoke(cli, ["plot", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "plots/test-results.png").is_file()
    stats = json.loads((tmp_path / "plots/test-statistics.json").read_text())
    assert stats["n"] == 1
    assert stats["mae_kcal_mol"] == pytest.approx(6.27509474)
    result = CliRunner().invoke(cli, ["plot", "--workdir", str(tmp_path)])
    assert result.exit_code != 0 and "already exist" in result.output


def test_named_project_completion_and_override(paired_databases, config_dir):
    source, _ = paired_databases
    root = config_dir / "ml"
    root.mkdir(parents=True)
    (root / "electronic.yaml").write_text(
        "data:\n  filename: ../../source.db\n"
        "  reference: ../../reference.db\n"
        "train:\n  pairing: revqm9-index\n  model: ridge\n"
        "  output: runs/test\n"
    )
    calls = []
    cli = recording_cli(calls)
    result = CliRunner().invoke(
        cli, ["-p", "electronic", "train", "--model", "linear"]
    )
    assert result.exit_code == 0, result.output
    options = calls[0][1]
    assert options["model"] == "linear" and options["filename"] == str(source)
    assert [item.value for item in complete_project(None, None, "ele")] == [
        "electronic"
    ]
    result = CliRunner().invoke(cli, ["-p", "missing", "train"])
    assert result.exit_code != 0 and "not found" in result.output


def test_default_output_is_invocation_directory(
    paired_databases, tmp_path, monkeypatch
):
    source, reference = paired_databases
    presets = tmp_path / "presets"
    presets.mkdir()
    project = presets / "test.yaml"
    project.write_text("train:\n  pairing: revqm9-index\n")
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    calls = []
    cli = recording_cli(calls)
    runner = CliRunner()
    args = [
        "-p",
        str(project),
        "-f",
        str(source),
        "-r",
        str(reference),
        "train",
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    first = Path(calls[-1][1]["output"])
    assert first.parent == working and first.name.startswith("ml-train-")
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert calls[-1][1]["output"] != str(first)
    result = runner.invoke(cli, args + ["-o", "chosen"])
    assert result.exit_code == 0, result.output
    assert calls[-1][1]["output"] == str(working / "chosen")
