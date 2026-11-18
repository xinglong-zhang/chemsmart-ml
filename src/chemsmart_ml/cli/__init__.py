"""Command-line interface for standalone and CHEMSMART-hosted use."""

import logging
import sys

from .ml import build_cli, execute

cli = build_cli(execute)


def main(*args, **kwargs):
    """Standalone entry point; reports progress on standard error."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stderr
    )
    return cli(*args, **kwargs)


__all__ = ["build_cli", "cli", "execute", "main"]
