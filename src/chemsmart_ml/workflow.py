"""Reproducible linear delta baseline; JSON model artifacts avoid pickle loading."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_database, pair


def features(row, elements, electronic):
    values = [row["atoms"].count(z) for z in elements] + [row["energy"]]
    if electronic:
        keys = ("homo", "lumo") if electronic == "orbitals" else ("homo", "lumo", "dipole")
        values += [row[k] for k in keys]
    if any(x is None or not np.isfinite(x) for x in values):
        names = list(elements) + ["energy"] + (list(keys) if electronic else [])
        missing = [name for name,value in zip(names,values) if value is None or not np.isfinite(value)]
        raise ValueError(f"Missing/nonfinite input features {missing} for record {row.get('record_id')}; inspect feature-availability.json")
    if set(row["atoms"])-set(elements):
        raise ValueError("Prediction contains elements absent from training")
    return values


def groups(pairs):
    # Connected components prevent either endpoint identity crossing splits.
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, a, b in pairs:
        x,y = root(a["molecule_id"]), root(b["molecule_id"])
        parent[max(x,y)] = min(x,y)
    return [root(a["molecule_id"]) for _,a,_ in pairs]


def train(filename, reference, output, seed=42, electronic=False, model="linear", alpha=1.0, fingerprint="none", morgan_radius=2, morgan_size=2048, morgan_chirality=False, cache=None, manifest=None, plot=True, train_fraction=0.8, valid_fraction=0.1, test_fraction=0.1, descriptor="none", soap_r_cut=5.0, soap_n_max=2, soap_l_max=2, soap_sigma=0.5, soap_aggregation="sum", soap_cache=None, max_iter=100, learning_rate=0.1, max_leaf_nodes=31, hidden_size=128, kernel_alpha=0.001, kernel_gamma=0.01, kernel_max_samples=2000, kernel_subsample=False):
    if model not in ("linear", "ridge", "hist-gradient-boosting", "mlp", "kernel-ridge") or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid model or regularization strength")
    fractions=np.array([train_fraction,valid_fraction,test_fraction],dtype=float)
    if not np.isfinite(fractions).all() or (fractions<=0).any() or not np.isclose(fractions.sum(),1.0,rtol=0,atol=1e-9):
        raise ValueError('Split fractions must be positive and sum to 1')
    if descriptor not in ('none','soap'):
        raise ValueError('Unsupported 3D descriptor')
    if plot:
        try:
            __import__("matplotlib")  # Check before the expensive database read.
        except ImportError as exc:
            raise ValueError('Automatic plotting requires matplotlib; install chemsmart-ml[plot] or use --no-plot') from exc
    model_name = model
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    pairs, report = pair(read_database(filename,"source", include_smiles=fingerprint=="morgan", include_positions=descriptor=="soap"), read_database(reference,"reference"))
    group_by_index = dict(zip([key for key,_,_ in pairs],groups(pairs)))
    if manifest:
        from .validation import load_manifest
        selection=load_manifest(manifest,filename,reference)
        (out/'validation-manifest.json').write_text(json.dumps(selection,indent=2))
        selected={item['npz_index']:item for item in selection['samples']}
        for key,a,b in pairs:
            if key in selected and (a['record_id']!=selected[key]['source_record'] or b['record_id']!=selected[key]['reference_record']):
                raise ValueError('Sample IDs differ from validation manifest')
        pairs=[row for row in pairs if row[0] in selected]
        if len(pairs)!=len(selected):
            raise ValueError('Validation manifest contains unmatched samples')
        report['selected']=len(pairs)
        report['validation_manifest']=str(Path(manifest).resolve())
        report['excluded_by_manifest']=report['matched']-len(pairs)
    (out/"pairing.json").write_text(json.dumps(report, indent=2))
    if report["rejected"] and not manifest:
        raise ValueError("Pairing rejected composition mismatches; inspect pairing.json and establish a verified index mapping before training")
    if len(pairs)<10:
        raise ValueError("Need at least 10 paired samples")
    availability = {key: sum(bool(a[key] is not None and np.isfinite(a[key])) for _,a,_ in pairs)
                    for key in ("energy", "homo", "lumo", "dipole")}
    (out/"feature-availability.json").write_text(json.dumps(dict(samples=len(pairs),finite_counts=availability),indent=2))
    elements = sorted({z for _,a,_ in pairs for z in a["atoms"]})
    x = np.array([features(a,elements,electronic) for _,a,_ in pairs], dtype=float)
    fingerprint_settings = None
    if fingerprint == "morgan":
        from .fingerprints import morgan_config, cached_morgan
        fingerprint_settings = morgan_config(morgan_radius, morgan_size, morgan_chirality)
        try:
            vectors, cache_report = cached_morgan([a for _,a,_ in pairs],fingerprint_settings,cache)
        except ValueError as exc:
            (out/'fingerprint-error.json').write_text(json.dumps(dict(error=str(exc),settings=fingerprint_settings),indent=2))
            raise
        x = np.column_stack([x,vectors])
        (out/'fingerprints.json').write_text(json.dumps(cache_report,indent=2))
    elif fingerprint != "none":
        raise ValueError('Unsupported fingerprint')
    soap_settings = None
    if descriptor == "soap":
        from .soap_features import soap_config,cached_soap
        soap_settings=soap_config(elements,soap_r_cut,soap_n_max,soap_l_max,soap_sigma,soap_aggregation)
        vectors,soap_report=cached_soap([a for _,a,_ in pairs],soap_settings,soap_cache)
        x=np.column_stack([x,vectors])
        (out/'soap.json').write_text(json.dumps(soap_report,indent=2))
    low = np.array([a["energy"] for _,a,_ in pairs])
    target = np.array([b["energy"] for _,_,b in pairs], dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Invalid reference energies")
    labels = []
    for group in [group_by_index[key] for key,_,_ in pairs]:
        h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12],16)/16**12
        labels.append("train" if h<train_fraction else "valid" if h<(train_fraction+valid_fraction) else "test")
    labels = np.array(labels, dtype="<U6")
    if any(not (labels==s).any() for s in ("train","valid","test")):
        raise ValueError("Empty split; need more independent molecular groups")
    if model_name == 'kernel-ridge' and kernel_subsample:
        candidates=np.flatnonzero(labels=='train')
        if len(candidates)>kernel_max_samples:
            selected=np.random.default_rng(seed).choice(candidates,size=kernel_max_samples,replace=False)
            labels[candidates]='unused'
            labels[selected]='train'
    mask = labels=="train"
    mean, scale = x[mask].mean(0), x[mask].std(0)
    scale[scale==0]=1
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    tree_model = None
    nonlinear_model = None
    if model_name in ("mlp", "kernel-ridge"):
        from .regression_models import fit_nonlinear,predict_nonlinear
        base_size=len(elements)+2
        base_coef=np.linalg.lstsq(design[mask,:base_size],(target-low)[mask],rcond=None)[0]
        residual=(target-low)[mask]-design[mask,:base_size]@base_coef
        nonlinear_model=fit_nonlinear(model_name,design[mask,1:],residual,seed,hidden_size,max_iter,learning_rate,kernel_alpha,kernel_gamma,kernel_max_samples)
        nonlinear_model['base_coefficients']=base_coef.tolist()
        nonlinear_model['kernel_subsample']=bool(kernel_subsample) if model_name=='kernel-ridge' else None
        correction=design[:,:base_size]@base_coef+predict_nonlinear(design[:,1:],nonlinear_model)
    elif model_name == "hist-gradient-boosting":
        from .tree_model import fit_tree,predict_tree
        # Retain an unregularized composition/energy baseline to remove the
        # extensive elemental offset before learning nonlinear corrections.
        base_size=len(elements)+2
        base_coef=np.linalg.lstsq(design[mask,:base_size],(target-low)[mask],rcond=None)[0]
        residual=(target-low)[mask]-design[mask,:base_size]@base_coef
        tree_model=fit_tree(design[mask,1:],residual,seed,max_iter,learning_rate,max_leaf_nodes)
        tree_model['base_coefficients']=base_coef.tolist()
        correction=design[:,:base_size]@base_coef+predict_tree(design[:,1:],tree_model)
    elif model_name == "ridge" and alpha > 0:
        # Avoid an expensive tall-matrix SVD and extra full training-matrix copy.
        gram = np.eye(design.shape[1]) * alpha
        gram[0,0] = 0
        rhs = np.zeros(design.shape[1])
        indices = np.flatnonzero(mask)
        delta = target-low
        for start in range(0,len(indices),2048):
            selected = indices[start:start+2048]
            block = design[selected]
            gram += block.T @ block
            rhs += block.T @ delta[selected]
        coef = np.linalg.solve(gram,rhs)
    else:
        coef = np.linalg.lstsq(design[mask], (target-low)[mask], rcond=None)[0]
    if tree_model is None and nonlinear_model is None:
        correction=design@coef
    pred = low + correction
    model = dict(format_version=1, model=model_name+"-delta", alpha=alpha if tree_model is None and nonlinear_model is None else None, target="total_energy", unit="Hartree",
                 elements=elements, electronic=electronic, fingerprint=fingerprint_settings, mean=mean.tolist(), scale=scale.tolist(),
                 coefficients=coef.tolist() if tree_model is None and nonlinear_model is None else None, tree=tree_model, nonlinear=nonlinear_model, soap=soap_settings, split=dict(train=train_fraction,valid=valid_fraction,test=test_fraction), seed=seed, plot=plot,
                 source=str(Path(filename).resolve()), reference=str(Path(reference).resolve()))
    (out/"model.json").write_text(json.dumps(model,indent=2))
    metrics = {s:dict(n=int((labels==s).sum()),mae_hartree=float(np.abs(pred[labels==s]-target[labels==s]).mean())) for s in ("train","valid","test")}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    with (out/"samples.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["npz_index","split","source_record","reference_record","reference_energy","prediction"])
        for i,(key,a,b) in enumerate(pairs):
            w.writerow([key,labels[i],a["record_id"],b["record_id"],target[i],pred[i]])
    if plot:
        from .plotting import plot_results
        plot_results(out,split='test')
    return metrics


def predict(filename, model_path, output):
    model=json.loads(Path(model_path).read_text())
    if model.get("format_version")!=1 or model.get("model") not in ("linear-delta", "ridge-delta", "hist-gradient-boosting-delta", "mlp-delta", "kernel-ridge-delta"):
        raise ValueError("Unsupported model format")
    settings = model.get('fingerprint')
    soap_settings=model.get('soap')
    rows=read_database(filename,"source",include_smiles=settings is not None,include_positions=soap_settings is not None)
    vectors = None
    if settings:
        from .fingerprints import cached_morgan
        vectors, _ = cached_morgan([row for _,row in sorted(rows.items())],settings)

    soap_vectors=None
    if soap_settings:
        from .soap_features import cached_soap
        soap_vectors,_=cached_soap([row for _,row in sorted(rows.items())],soap_settings)
    ordered=sorted(rows.items())
    with Path(output).open("x") as f:
        w=csv.writer(f);w.writerow(["npz_index","record_id","predicted_energy_hartree"])
        for start in range(0,len(ordered),1024):
            batch=ordered[start:start+1024]
            x=np.asarray([features(row,model["elements"],model["electronic"]) for _,row in batch])
            if vectors is not None:
                x=np.column_stack([x,vectors[start:start+len(batch)]])
            if soap_vectors is not None:
                x=np.column_stack([x,soap_vectors[start:start+len(batch)]])
            design=np.column_stack([np.ones(len(batch)),(x-model["mean"])/model["scale"]])
            if model['model'] in ('mlp-delta','kernel-ridge-delta'):
                from .regression_models import predict_nonlinear
                nonlinear=model['nonlinear'];coef=nonlinear['base_coefficients']
                correction=design[:,:len(coef)]@coef+predict_nonlinear(design[:,1:],nonlinear)
            elif model['model']=='hist-gradient-boosting-delta':
                from .tree_model import predict_tree
                tree=model['tree'];coef=tree['base_coefficients']
                correction=design[:,:len(coef)]@coef+predict_tree(design[:,1:],tree)
            else:
                correction=design@model['coefficients']
            for (key,row),value in zip(batch,correction):
                w.writerow([key,row['record_id'],row['energy']+value])
    return dict(samples=len(rows),output=str(output))


def prepare(filename, output, morgan_radius=2, morgan_size=2048, morgan_chirality=False, cache=None, manifest=None):
    """Populate reusable Morgan cache without labels, splitting or training."""
    from .fingerprints import morgan_config, cached_morgan
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    settings = morgan_config(morgan_radius,morgan_size,morgan_chirality)
    rows = read_database(filename,'source',include_smiles=True)
    if manifest:
        from .validation import load_manifest
        selection=load_manifest(manifest,filename)
        selected={item['npz_index']:item for item in selection['samples']}
        rows={key:row for key,row in rows.items() if key in selected}
        if len(rows)!=len(selected) or any(row['record_id']!=selected[key]['source_record'] for key,row in rows.items()):
            raise ValueError('Source sample IDs differ from manifest')
    try:
        _, report = cached_morgan(list(rows.values()),settings,cache)
    except ValueError as exc:
        (out/'fingerprint-error.json').write_text(json.dumps(dict(error=str(exc),settings=settings),indent=2))
        raise
    (out/'fingerprints.json').write_text(json.dumps(report,indent=2))
    return report


def validate(**options):
    from .validation import validate as validate_database
    return validate_database(**options)
