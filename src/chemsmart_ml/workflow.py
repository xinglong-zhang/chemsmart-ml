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


def train(filename, reference, output, seed=42, electronic=False, model="linear", alpha=1.0, fingerprint="none", morgan_radius=2, morgan_size=2048, morgan_chirality=False, cache=None):
    if model not in ("linear", "ridge") or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid model or regularization strength")
    model_name = model
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    pairs, report = pair(read_database(filename,"source", include_smiles=fingerprint=="morgan"), read_database(reference,"reference"))
    (out/"pairing.json").write_text(json.dumps(report, indent=2))
    if report["rejected"]:
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
    low = np.array([a["energy"] for _,a,_ in pairs])
    target = np.array([b["energy"] for _,_,b in pairs], dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Invalid reference energies")
    labels = []
    for group in groups(pairs):
        h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12],16)/16**12
        labels.append("train" if h<0.8 else "valid" if h<0.9 else "test")
    labels = np.array(labels)
    if any(not (labels==s).any() for s in ("train","valid","test")):
        raise ValueError("Empty split; need more independent molecular groups")
    mask = labels=="train"
    mean, scale = x[mask].mean(0), x[mask].std(0)
    scale[scale==0]=1
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    if model_name == "ridge" and alpha > 0:
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
    pred = low + design@coef
    model = dict(format_version=1, model=model_name+"-delta", alpha=alpha, target="total_energy", unit="Hartree",
                 elements=elements, electronic=electronic, fingerprint=fingerprint_settings, mean=mean.tolist(), scale=scale.tolist(),
                 coefficients=coef.tolist(), seed=seed,
                 source=str(Path(filename).resolve()), reference=str(Path(reference).resolve()))
    (out/"model.json").write_text(json.dumps(model,indent=2))
    metrics = {s:dict(n=int((labels==s).sum()),mae_hartree=float(np.abs(pred[labels==s]-target[labels==s]).mean())) for s in ("train","valid","test")}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    with (out/"samples.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["npz_index","split","source_record","reference_record","reference_energy","prediction"])
        for i,(key,a,b) in enumerate(pairs):
            w.writerow([key,labels[i],a["record_id"],b["record_id"],target[i],pred[i]])
    return metrics


def predict(filename, model_path, output):
    model=json.loads(Path(model_path).read_text())
    if model.get("format_version")!=1 or model.get("model") not in ("linear-delta", "ridge-delta"):
        raise ValueError("Unsupported model format")
    settings = model.get('fingerprint')
    rows=read_database(filename,"source",include_smiles=settings is not None)
    vectors = None
    if settings:
        from .fingerprints import cached_morgan
        vectors, _ = cached_morgan([row for _,row in sorted(rows.items())],settings)

    with Path(output).open("x") as f:
        w=csv.writer(f);w.writerow(["npz_index","record_id","predicted_energy_hartree"])
        for i,(key,row) in enumerate(sorted(rows.items())):
            x=np.array(features(row,model["elements"],model["electronic"]))
            if vectors is not None:
                x = np.r_[x,vectors[i]]
            d=np.r_[1,(x-model["mean"])/model["scale"]]
            w.writerow([key,row["record_id"],row["energy"]+d@model["coefficients"]])
    return dict(samples=len(rows),output=str(output))


def prepare(filename, output, morgan_radius=2, morgan_size=2048, morgan_chirality=False, cache=None):
    """Populate reusable Morgan cache without labels, splitting or training."""
    from .fingerprints import morgan_config, cached_morgan
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    settings = morgan_config(morgan_radius,morgan_size,morgan_chirality)
    rows = read_database(filename,'source',include_smiles=True)
    try:
        _, report = cached_morgan(list(rows.values()),settings,cache)
    except ValueError as exc:
        (out/'fingerprint-error.json').write_text(json.dumps(dict(error=str(exc),settings=settings),indent=2))
        raise
    (out/'fingerprints.json').write_text(json.dumps(report,indent=2))
    return report
