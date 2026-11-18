"""Reusable click option groups and callbacks for ``chemsmart-ml``."""

import uuid
from pathlib import Path

import click
from click.shell_completion import CompletionItem

from chemsmart_ml.features.builder import FeatureBuilder
from chemsmart_ml.models.delta import DeltaModel
from chemsmart_ml.projects import ProjectConfig

FEATURE_SETS = list(FeatureBuilder.ELECTRONIC_KEYS)
PAIRING_HELP = (
    "xTB idx=N matches NPZ structure_index_in_file=N. Both inputs must "
    "share this dataset ordering."
)


def resolve_project(ctx, param, value):
    """Click callback resolving a project name or YAML path."""
    if value is None:
        return None
    try:
        return str(ProjectConfig.resolve(value))
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def complete_project(ctx, param, incomplete):
    """Shell completion of named projects."""
    return [
        CompletionItem(name)
        for name in ProjectConfig.available_names()
        if name.startswith(incomplete)
    ]


def _apply(f, options):
    """Apply click options so they appear in list order in --help."""
    for option in reversed(options):
        f = option(f)
    return f


def click_feature_options(features="composition-energy", fingerprint="none"):
    """Feature set, fingerprint and pairing convention options."""

    def decorator(f):
        return _apply(
            f,
            [
                click.option(
                    "--pairing",
                    type=click.Choice(["revqm9-index"]),
                    required=True,
                    help=PAIRING_HELP,
                ),
                click.option(
                    "--features",
                    type=click.Choice(FEATURE_SETS),
                    default=features,
                    show_default=True,
                ),
                click.option(
                    "--fingerprint",
                    type=click.Choice(["none", "morgan"]),
                    default=fingerprint,
                    show_default=True,
                ),
            ],
        )

    return decorator


def click_morgan_options(f):
    """Morgan fingerprint, cache and manifest options."""

    return _apply(
        f,
        [
            click.option(
                "--morgan-radius", type=click.IntRange(min=0), default=2
            ),
            click.option(
                "--morgan-size", type=click.IntRange(min=1), default=2048
            ),
            click.option(
                "--morgan-chirality/--no-morgan-chirality", default=False
            ),
            click.option(
                "--manifest",
                type=click.Path(exists=True, resolve_path=True),
                help="Validation manifest (valid-samples.json) restricting "
                "the samples used.",
            ),
            click.option(
                "--cache",
                type=click.Path(resolve_path=True),
                help="Morgan SQLite cache file.",
            ),
        ],
    )


def click_split_options(f):
    """Train/validation/test fraction options."""

    fraction = click.FloatRange(min=0, max=1)
    return _apply(
        f,
        [
            click.option(
                f"--{name}-fraction",
                type=fraction,
                default=default,
                show_default=True,
            )
            for name, default in [
                ("train", 0.8),
                ("valid", 0.1),
                ("test", 0.1),
            ]
        ],
    )


def click_soap_options(f):
    """SOAP descriptor options."""

    return _apply(
        f,
        [
            click.option(
                "--descriptor",
                type=click.Choice(["none", "soap"]),
                default="none",
            ),
            click.option(
                "--soap-r-cut",
                type=click.FloatRange(min=1, min_open=True),
                default=5.0,
            ),
            click.option(
                "--soap-n-max", type=click.IntRange(min=1), default=2
            ),
            click.option(
                "--soap-l-max", type=click.IntRange(min=0, max=20), default=2
            ),
            click.option(
                "--soap-sigma",
                type=click.FloatRange(min=0, min_open=True),
                default=0.5,
            ),
            click.option(
                "--soap-aggregation",
                type=click.Choice(["sum", "mean"]),
                default="sum",
            ),
            click.option(
                "--soap-cache",
                type=click.Path(resolve_path=True),
                help="SOAP SQLite cache file.",
            ),
        ],
    )


def click_model_options(f):
    """Model choice and hyperparameter options."""

    return _apply(
        f,
        [
            click.option(
                "--model",
                type=click.Choice(DeltaModel.names()),
                default="linear",
                show_default=True,
            ),
            click.option("--alpha", default=1.0, type=click.FloatRange(min=0)),
            click.option(
                "--hidden-size", type=click.IntRange(min=1), default=128
            ),
            click.option(
                "--kernel-alpha",
                type=click.FloatRange(min=0, min_open=True),
                default=0.001,
            ),
            click.option(
                "--kernel-gamma",
                type=click.FloatRange(min=0, min_open=True),
                default=0.01,
            ),
            click.option(
                "--kernel-subsample/--no-kernel-subsample",
                default=False,
                help="Explicitly limit kernel training samples; "
                "validation/test remain unchanged.",
            ),
            click.option(
                "--kernel-max-samples",
                type=click.IntRange(min=1, max=10000),
                default=2000,
            ),
            click.option(
                "--cv-folds",
                type=click.IntRange(min=0),
                default=0,
                help="Grouped CV folds on training+validation, holding the "
                "test set out; 0 disables CV.",
            ),
            click.option(
                "--max-iter", type=click.IntRange(min=1), default=100
            ),
            click.option(
                "--learning-rate",
                type=click.FloatRange(min=0, min_open=True),
                default=None,
                help="Defaults to 0.001 for mlp and 0.1 otherwise.",
            ),
            click.option(
                "--max-leaf-nodes", type=click.IntRange(min=2), default=31
            ),
        ],
    )


def default_train_output():
    """New ``ml-train-*`` directory under the working directory."""
    return str(Path.cwd() / ("ml-train-" + uuid.uuid4().hex[:8]))
