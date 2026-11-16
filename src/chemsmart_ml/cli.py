"""Standalone and host-shareable command definitions."""
import json
import click


def build_cli(dispatch):
    @click.group()
    @click.option('-f','--filename', required=True, type=click.Path(exists=True,resolve_path=True))
    @click.option('-r','--reference', type=click.Path(exists=True,resolve_path=True))
    @click.pass_context
    def main(ctx, filename, reference):
        """Learn DFT energies from paired CHEMSMART xTB databases."""
        ctx.ensure_object(dict)
        ctx.obj['ml_input'] = dict(filename=filename, reference=reference)

    @main.command()
    @click.option('--target',type=click.Choice(['energy']),default='energy')
    @click.option('--learning',type=click.Choice(['delta']),default='delta')
    @click.option('--pairing',type=click.Choice(['revqm9-index']),required=True,
                  help='xTB idx=N matches NPZ inds=N-1. Both inputs must share this dataset ordering.')
    @click.option('--features',type=click.Choice(['composition-energy','electronic']),default='composition-energy')
    @click.option('--seed',default=42,type=int)
    @click.option('-o','--output',required=True,type=click.Path(resolve_path=True))
    @click.pass_context
    def train(ctx,target,learning,pairing,features,seed,output):
        options=dict(ctx.obj['ml_input'])
        if options['reference'] is None:
            raise click.UsageError('Training requires -r/--reference')
        options.update(output=output,seed=seed,electronic=features=='electronic')
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
