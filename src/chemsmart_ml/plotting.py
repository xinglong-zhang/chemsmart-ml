"""Plot saved predictions without reading training databases or loading models."""
import csv
import json
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL_MOL = 627.509474


def plot_results(workdir, split='test', output=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError('Plotting requires matplotlib: install chemsmart-ml[plot]') from exc
    root = Path(workdir)
    with (root/'samples.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'split','reference_energy','prediction'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('samples.csv lacks split/reference_energy/prediction columns')
        rows = [row for row in reader if row['split']==split]
    if not rows:
        raise ValueError(f'No samples for split {split}')
    reference = np.array([float(row['reference_energy']) for row in rows])
    prediction = np.array([float(row['prediction']) for row in rows])
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError('Predictions contain nonfinite values')
    error = (prediction-reference)*HARTREE_TO_KCAL_MOL
    stats = dict(split=split, n=len(rows), mae_kcal_mol=float(abs(error).mean()),
                 rmse_kcal_mol=float(np.sqrt(np.mean(error**2))),
                 bias_kcal_mol=float(error.mean()),
                 p95_absolute_error_kcal_mol=float(np.percentile(abs(error),95)))
    out = Path(output) if output else root/'plots'
    out.mkdir(parents=True,exist_ok=True)
    paths = [out/f'{split}-results.png',out/f'{split}-results.pdf',out/f'{split}-statistics.json']
    if any(path.exists() for path in paths):
        raise ValueError('Plot files already exist; choose a new --output directory')
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    try:
        axes[0].hexbin(reference,prediction,gridsize=65,mincnt=1,bins='log')
        lo,hi = min(reference.min(),prediction.min()),max(reference.max(),prediction.max())
        axes[0].plot([lo,hi],[lo,hi],color='coral')
        axes[0].set(xlabel='Reference DFT energy (Hartree)',ylabel='Predicted DFT energy (Hartree)',title='Prediction vs reference')
        axes[1].hist(error,bins=80,color='steelblue')
        axes[1].axvline(0,color='coral')
        axes[1].set(xlabel='Prediction minus reference (kcal/mol)',ylabel='Samples',title=f"MAE = {stats['mae_kcal_mol']:.2f} kcal/mol")
        axes[2].hexbin(reference,error,gridsize=60,mincnt=1,bins='log')
        axes[2].axhline(0,color='coral')
        axes[2].set(xlabel='Reference DFT energy (Hartree)',ylabel='Error (kcal/mol)',title='Errors across energy range')
        fig.suptitle(f'{split.capitalize()} predictions | {len(rows):,} samples')
        for path in paths[:2]:
            fig.savefig(path,dpi=170)
    finally:
        plt.close(fig)
    paths[2].write_text(json.dumps(stats,indent=2))
    return dict(statistics=stats,files=[str(path.resolve()) for path in paths])
