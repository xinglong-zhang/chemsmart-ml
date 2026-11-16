import csv
import json
import numpy as np
import pytest
from click.testing import CliRunner
from chemsmart_ml.cli import build_cli, execute
from chemsmart_ml.regression_models import fit_nonlinear
from chemsmart_ml.workflow import predict
from test_workflow import database


def test_kernel_limit():
    x = np.arange(30).reshape(15, 2).astype(float)
    with pytest.raises(ValueError, match="memory scales quadratically"):
        fit_nonlinear("kernel-ridge", x, np.arange(15), 42, kernel_max_samples=10)


@pytest.mark.parametrize("name", ["mlp", "kernel-ridge"])
def test_training_and_portable_prediction(tmp_path, name):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    database(a)
    database(b, True)
    options = [
        "-f",
        str(a),
        "-r",
        str(b),
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
        labels = [r["split"] for r in csv.DictReader((tmp_path / "trained/samples.csv").open())]
        assert labels.count("train") == 50 and "unused" in labels
    output = tmp_path / "prediction.csv"
    predict(a, tmp_path / "trained/model.json", output)
    got = {
        r["record_id"]: float(r["predicted_energy_hartree"])
        for r in csv.DictReader(output.open())
    }
    for row in csv.DictReader((tmp_path / "trained/samples.csv").open()):
        assert got[row["source_record"]] == pytest.approx(
            float(row["prediction"]), abs=1e-8
        )
