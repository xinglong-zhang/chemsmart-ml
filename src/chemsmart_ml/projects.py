"""Named projects and shell completion using CHEMSMART's configuration root."""
import os
from pathlib import Path
import click
from click.shell_completion import CompletionItem


def project_directory():
    try:
        from chemsmart.settings.user import CHEMSMARTUserSettings
    except ImportError:
        root = os.environ.get('CHEMSMART_CONFIG_DIR', '~/.chemsmart')
    else:
        root = CHEMSMARTUserSettings.resolve_config_dir()
    return Path(root).expanduser()/'ml'


def resolve_project(ctx, param, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.parent == Path('.'):
        root = project_directory()
        candidates = [root/path] if path.suffix in ('.yaml','.yml') else [root/(value+suffix) for suffix in ('.yaml','.yml')]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
    raise click.BadParameter(f'Project {value!r} not found; use a YAML path or a name in {project_directory()}')


def complete_project(ctx, param, incomplete):
    root = project_directory()
    names = {p.stem for pattern in ('*.yaml','*.yml') for p in root.glob(pattern) if p.is_file()}
    return [CompletionItem(name) for name in sorted(names) if name.startswith(incomplete)]
