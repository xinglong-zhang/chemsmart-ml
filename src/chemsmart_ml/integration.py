"""CHEMSMART plugin adapter; the core workflows have no host dependency."""

import logging
from pathlib import Path

from chemsmart.jobs.job import Job
from chemsmart.jobs.runner import JobRunner
from chemsmart.settings.executable import Executable

from chemsmart_ml.cli import build_cli, execute

logger = logging.getLogger(__name__)


class MLExecutable(Executable):
    PROGRAM = "ML"


class MLJob(Job):
    """
    CHEMSMART job wrapping one ``chemsmart-ml`` workflow action.

    Attributes:
        action (str): Workflow action ('train', 'predict', ...).
        options (dict): Final command options for the workflow.
    """

    TYPE = "ml"
    PROGRAM = "ML"

    def __init__(self, action, options):
        self.action = action
        self.options = options
        super().__init__(
            molecule=None,
            label=Path(options["output"]).stem + "_" + action,
            jobrunner=None,
            skip_completed=False,
        )

    def is_complete(self):
        return False

    def _run(self, **kwargs):
        return self.jobrunner.run(self, **kwargs)


class MLJobRunner(JobRunner):
    """Runs ML jobs in-process on persistent (non-scratch) paths."""

    JOBTYPES = ["ml"]
    PROGRAM = "ML"
    SCRATCH = False
    FAKE = False

    def __init__(self, server=None, scratch=None, **kwargs):
        if scratch:
            raise ValueError("ML requires persistent paths; use --no-scratch")
        super().__init__(server=server, scratch=False, **kwargs)

    def run(self, job, **kwargs):
        if self.fake:
            raise ValueError("ML does not support fake execution")
        return execute(job.action, job.options)


ml = build_cli(MLJob)
ml.name = "ml"
ml.chemsmart_classes = (MLJob, MLJobRunner, MLExecutable)
