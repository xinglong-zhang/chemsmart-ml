"""Task-specific input screening and explicit, reusable sample manifests."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_database, pair


def identity(path):
    p=Path(path).resolve();stat=p.stat()
    return dict(path=str(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def load_manifest(path, filename, reference=None):
    data=json.loads(Path(path).read_text())
    if data.get('format_version')!=1 or data.get('source')!=identity(filename):
        raise ValueError('Validation manifest does not match the source database; rerun validate')
    if reference is not None and data.get('reference')!=identity(reference):
        raise ValueError('Validation manifest does not match the reference database; rerun validate')
    return data


def validate(filename, reference, output, electronic='orbitals', fingerprint='morgan'):
    if fingerprint not in ('none','morgan'):
        raise ValueError('Unsupported fingerprint')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    pairs, pairing=pair(read_database(filename,'source',include_smiles=fingerprint=='morgan'),
                        read_database(reference,'reference'))
    keys=['energy']
    if electronic:
        keys+=['homo','lumo'] if electronic=='orbitals' else ['homo','lumo','dipole']
    smiles_valid={}
    if fingerprint=='morgan':
        from rdkit import Chem, rdBase
    good=[];bad=[]
    for index,a,b in pairs:
        reasons=[]
        for key in keys:
            if a[key] is None or not np.isfinite(a[key]):
                reasons.append('missing/nonfinite source '+key)
        if b['energy'] is None or not np.isfinite(b['energy']):
            reasons.append('missing/nonfinite reference energy')
        if fingerprint=='morgan':
            smiles=a.get('smiles')
            if smiles not in smiles_valid:
                with rdBase.BlockLogs():
                    mol=Chem.MolFromSmiles(smiles) if smiles else None
                smiles_valid[smiles]=mol is not None
            if not smiles_valid[smiles]:
                reasons.append('missing/invalid RDKit SMILES')
        item=dict(npz_index=index,idx=index+1,source_record=a['record_id'],reference_record=b['record_id'])
        if reasons:
            bad.append(dict(**item,reasons='; '.join(reasons),smiles=a.get('smiles','')))
        else:
            good.append(item)
    for row in pairing['rejected']:
        bad.append(dict(npz_index=row['index'],idx=row['index']+1,source_record='',reference_record='',reasons=row['reason'],smiles=''))
    manifest=dict(format_version=1,source=identity(filename),reference=identity(reference),
                  electronic=electronic,fingerprint=fingerprint,samples=good)
    (out/'valid-samples.json').write_text(json.dumps(manifest,indent=2))
    with (out/'excluded.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['npz_index','idx','source_record','reference_record','reasons','smiles'])
        w.writeheader();w.writerows(bad)
    report=dict(valid=len(good),excluded=len(bad),pairing=pairing,
                scope='Pairing, required numeric features and RDKit SMILES parsing; not a geometry/convergence audit',
                manifest=str((out/'valid-samples.json').resolve()))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    return report
