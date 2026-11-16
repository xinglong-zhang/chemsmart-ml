"""CHEMSMART plugin API adapter; core workflow has no host dependency."""
from pathlib import Path
from chemsmart.jobs.job import Job
from chemsmart.jobs.runner import JobRunner
from chemsmart.settings.executable import Executable
from .cli import build_cli, execute


class MLExecutable(Executable):
    PROGRAM = 'ML'


class MLJob(Job):
    TYPE = 'ml'
    PROGRAM = 'ML'

    def __init__(self, action, options):
        self.action, self.options = action, options
        super().__init__(molecule=None,label=Path(options['output']).stem+'_'+action,
                         jobrunner=None,skip_completed=False)

    def is_complete(self):
        return False

    def _run(self, **kwargs):
        return self.jobrunner.run(self, **kwargs)


class MLJobRunner(JobRunner):
    JOBTYPES = ['ml']
    PROGRAM = 'ML'
    SCRATCH = False
    FAKE = False

    def __init__(self,server=None,scratch=None,**kwargs):
        if scratch:
            raise ValueError('ML requires persistent paths; use --no-scratch')
        super().__init__(server=server,scratch=False,**kwargs)

    def run(self,job,**kwargs):
        if self.fake:
            raise ValueError('ML does not support fake execution')
        return execute(job.action,job.options)


ml=build_cli(lambda action,options: MLJob(action,options))
ml.name='ml'
ml.chemsmart_classes=(MLJob,MLJobRunner,MLExecutable)
