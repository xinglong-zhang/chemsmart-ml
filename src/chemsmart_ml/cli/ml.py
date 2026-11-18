"""Standalone and host-shareable ``chemsmart-ml`` command definitions."""

import json
import logging
from pathlib import Path

import click
import yaml

from chemsmart_ml.cli.options import (
    click_feature_options,
    click_model_options,
    click_morgan_options,
    click_soap_options,
    click_split_options,
    complete_project,
    default_train_output,
    resolve_project,
)
from chemsmart_ml.projects import ProjectConfig

logger = logging.getLogger(__name__)


def build_cli(dispatch):
    """
    Build the command group.

    Args:
        dispatch (callable): Called as ``dispatch(action, options)`` with
            the final options of a train/predict/prepare/validate
            command; standalone execution runs the workflow, while the
            CHEMSMART host wraps it in a job.
    """

    @click.group()
    @click.option(
        "-p",
        "--project",
        type=str,
        callback=resolve_project,
        shell_complete=complete_project,
        help="Project name in ~/.chemsmart/ml or an explicit YAML path.",
    )
    @click.option(
        "-f",
        "--filename",
        required=False,
        type=click.Path(exists=True, resolve_path=True),
        help="Low-level (source) CHEMSMART database.",
    )
    @click.option(
        "-r",
        "--reference",
        type=click.Path(exists=True, resolve_path=True),
        help="Reference CHEMSMART database.",
    )
    @click.pass_context
    def main(ctx, project, filename, reference):
        """Learn DFT energies from paired CHEMSMART xTB databases."""
        ctx.ensure_object(dict)
        data = {}
        if project:
            config = _read_project(project)
            try:
                defaults = config.command_defaults
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc
            ctx.default_map = {**(ctx.default_map or {}), **defaults}
            data = config.data
        if filename is None:
            filename = data.get("filename")
        if reference is None:
            reference = data.get("reference")
        if filename is None and ctx.invoked_subcommand != "plot":
            raise click.UsageError(
                "Provide -f/--filename or data.filename in YAML"
            )
        for value in (filename, reference):
            if value is not None and not Path(value).is_file():
                raise click.UsageError(f"Database does not exist: {value}")
        ctx.obj["ml_input"] = dict(filename=filename, reference=reference)

    @main.command()
    @click.option("--target", type=click.Choice(["energy"]), default="energy")
    @click.option("--learning", type=click.Choice(["delta"]), default="delta")
    @click_feature_options()
    @click.option(
        "--plot/--no-plot",
        default=True,
        show_default=True,
        help="Generate test-set plots after training.",
    )
    @click_model_options
    @click_morgan_options
    @click_split_options
    @click_soap_options
    @click.option("--seed", default=42, type=int)
    @click.option(
        "-o",
        "--output",
        default=default_train_output,
        type=click.Path(resolve_path=True),
        help="Output directory; defaults to a new ml-train-* directory "
        "under the working directory.",
    )
    @click.pass_context
    def train(ctx, target, learning, pairing, **options):
        """Train a delta model on paired source/reference databases."""
        if ctx.obj["ml_input"]["reference"] is None:
            raise click.UsageError("Training requires -r/--reference")
        return dispatch("train", {**ctx.obj["ml_input"], **options})

    @main.command()
    @click.option(
        "--model",
        "model_path",
        required=True,
        type=click.Path(exists=True, resolve_path=True),
    )
    @click.option(
        "-o", "--output", required=True, type=click.Path(resolve_path=True)
    )
    @click.pass_context
    def predict(ctx, model_path, output):
        """Predict energies for a source database with a saved model."""
        if ctx.obj["ml_input"]["reference"]:
            raise click.UsageError(
                "Prediction does not accept a reference database"
            )
        return dispatch(
            "predict",
            dict(
                filename=ctx.obj["ml_input"]["filename"],
                model_path=model_path,
                output=output,
            ),
        )

    @main.command()
    @click.option(
        "--workdir",
        required=True,
        type=click.Path(exists=True, file_okay=False, resolve_path=True),
    )
    @click.option(
        "--split",
        type=click.Choice(["train", "valid", "test"]),
        default="test",
        show_default=True,
    )
    @click.option("-o", "--output", type=click.Path(resolve_path=True))
    @click.pass_context
    def plot(ctx, workdir, split, output):
        """Plot saved results locally; no databases or retraining needed."""
        parent = ctx.parent
        while parent:
            if parent.info_name == "sub":
                raise click.UsageError(
                    "Plot is local; use chemsmart run ml plot"
                )
            parent = parent.parent
        try:
            result = _plot_results(Path(workdir), split, output)
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result, indent=2))

    @main.command()
    @click_morgan_options
    @click.option(
        "-o", "--output", required=True, type=click.Path(resolve_path=True)
    )
    @click.pass_context
    def prepare(ctx, **options):
        """Prepare reusable Morgan fingerprints without training."""
        return dispatch(
            "prepare",
            dict(filename=ctx.obj["ml_input"]["filename"], **options),
        )

    @main.command()
    @click_feature_options(features="orbitals", fingerprint="morgan")
    @click.option(
        "-o", "--output", required=True, type=click.Path(resolve_path=True)
    )
    @click.pass_context
    def validate(ctx, pairing, **options):
        """Screen paired samples and write a reusable manifest."""
        if not ctx.obj["ml_input"]["reference"]:
            raise click.UsageError("Validation requires -r/--reference")
        return dispatch("validate", {**ctx.obj["ml_input"], **options})

    return main


def execute(action, options):
    """Run a workflow in this process and print its JSON summary."""
    from chemsmart_ml.workflows import MLWorkflow

    try:
        result = MLWorkflow.from_action(action, options).run()
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))
    if action == "train":
        click.echo(f"Results: {options['output']}", err=True)
    return result


def _read_project(project) -> ProjectConfig:
    try:
        return ProjectConfig.from_file(project)
    except (OSError, yaml.YAMLError) as exc:
        raise click.ClickException(f"Cannot read project YAML: {exc}") from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _plot_results(workdir: Path, split, output):
    """Plot cross-validation folds if present, else saved predictions."""
    from chemsmart_ml.plotting import (
        CrossValidationPlotter,
        PredictionResultsPlotter,
    )

    report = workdir / CrossValidationPlotter.REPORT_FILENAME
    if report.is_file():
        if split != "test":
            raise ValueError(
                "Cross-validation plots show folds; --split does not apply"
            )
        destination = Path(output) if output else workdir / "plots"
        return CrossValidationPlotter.from_file(report).plot(destination)
    return PredictionResultsPlotter(workdir, split).plot(output)
