"""Standalone and host-shareable command definitions."""
import json
from pathlib import Path
import yaml
import click


def build_cli(dispatch):
    @click.group()
    @click.option('-p','--project', type=click.Path(exists=True,resolve_path=True))
    @click.option('-f','--filename', required=False, type=click.Path(exists=True,resolve_path=True))
    @click.option('-r','--reference', type=click.Path(exists=True,resolve_path=True))
    @click.pass_context
    def main(ctx, project, filename, reference):
        """Learn DFT energies from paired CHEMSMART xTB databases."""
        ctx.ensure_object(dict)
        config = {}
        if project:
            try:
                config = yaml.safe_load(Path(project).read_text())
            except (OSError, yaml.YAMLError) as exc:
                raise click.ClickException(f"Cannot read project YAML: {exc}") from exc
            if not isinstance(config, dict) or set(config) - {'data', 'train', 'predict'}:
                raise click.UsageError('Project YAML must contain only data/train/predict mappings')
            allowed = {
                'data': {'filename', 'reference'},
                'train': {'target','learning','pairing','features','model','alpha','seed','output'},
                'predict': {'model_path','output'},
            }
            for section, keys in allowed.items():
                values = config.get(section, {})
                if not isinstance(values, dict) or set(values)-keys:
                    raise click.UsageError(f'Unknown keys or invalid mapping in {section}')
                for key in ('filename','reference','output','model_path'):
                    if key in values:
                        if not isinstance(values[key], str):
                            raise click.UsageError(f'{section}.{key} must be a path string')
                        path = Path(values[key]).expanduser()
                        values[key] = str((Path(project).parent/path).resolve())
            ctx.default_map = {**(ctx.default_map or {}),
                               'train': config.get('train', {}),
                               'predict': config.get('predict', {})}
        data = config.get('data', {})
        filename = filename if filename is not None else data.get('filename')
        reference = reference if reference is not None else data.get('reference')
        if filename is None:
            raise click.UsageError('Provide -f/--filename or data.filename in YAML')
        for value in (filename, reference):
            if value is not None and not Path(value).is_file():
                raise click.UsageError(f'Database does not exist: {value}')
        ctx.obj['ml_input'] = dict(filename=filename, reference=reference)

    @main.command()
    @click.option('--target',type=click.Choice(['energy']),default='energy')
    @click.option('--learning',type=click.Choice(['delta']),default='delta')
    @click.option('--pairing',type=click.Choice(['revqm9-index']),required=True,
                  help='xTB idx=N matches NPZ structure_index_in_file=N. Both inputs must share this dataset ordering.')
    @click.option('--features',type=click.Choice(['composition-energy','electronic']),default='composition-energy')
    @click.option('--model',type=click.Choice(['linear','ridge']),default='linear',show_default=True)
    @click.option('--alpha',default=1.0,type=click.FloatRange(min=0))
    @click.option('--seed',default=42,type=int)
    @click.option('-o','--output',required=True,type=click.Path(resolve_path=True))
    @click.pass_context
    def train(ctx,target,learning,pairing,features,model,alpha,seed,output):
        options=dict(ctx.obj['ml_input'])
        if options['reference'] is None:
            raise click.UsageError('Training requires -r/--reference')
        options.update(output=output,seed=seed,electronic=features=='electronic',model=model,alpha=alpha)
        return dispatch('train',options)

    @main.command()
    @click.option('--model', 'model_path', required=True,type=click.Path(exists=True,resolve_path=True))
    @click.option('-o','--output',required=True,type=click.Path(resolve_path=True))
    @click.pass_context
    def predict(ctx,model_path,output):
        if ctx.obj['ml_input']['reference']:
            raise click.UsageError('Prediction does not accept a reference database')
        return dispatch('predict',dict(filename=ctx.obj['ml_input']['filename'],model_path=model_path,output=output))
    return main


def execute(action,options):
    from . import workflow
    try:
        result=getattr(workflow,action)(**options)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result,indent=2))
    return result

main=build_cli(execute)
