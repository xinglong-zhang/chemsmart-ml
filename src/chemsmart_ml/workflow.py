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
        values += [row[k] for k in ("homo", "lumo", "dipole")]
    if any(x is None or not np.isfinite(x) for x in values):
        raise ValueError("Missing/nonfinite input features; use --features composition-energy if needed")
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


def train(filename, reference, output, seed=42, electronic=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    pairs, report = pair(read_database(filename,"source"), read_database(reference,"reference"))
    (out/"pairing.json").write_text(json.dumps(report, indent=2))
    if len(pairs)<10:
        raise ValueError("Need at least 10 paired samples")
    elements = sorted({z for _,a,_ in pairs for z in a["atoms"]})
    x = np.array([features(a,elements,electronic) for _,a,_ in pairs], dtype=float)
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
    coef = np.linalg.lstsq(design[mask], (target-low)[mask], rcond=None)[0]
    pred = low + design@coef
    model = dict(format_version=1, model="linear-delta", target="total_energy", unit="Hartree",
                 elements=elements, electronic=electronic, mean=mean.tolist(), scale=scale.tolist(),
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
    if model.get("format_version")!=1 or model.get("model")!="linear-delta":
        raise ValueError("Unsupported model format")
    rows=read_database(filename,"source")
    with Path(output).open("x") as f:
        w=csv.writer(f);w.writerow(["npz_index","record_id","predicted_energy_hartree"])
        for key,row in sorted(rows.items()):
            x=np.array(features(row,model["elements"],model["electronic"]))
            d=np.r_[1,(x-model["mean"])/model["scale"]]
            w.writerow([key,row["record_id"],row["energy"]+d@model["coefficients"]])
    return dict(samples=len(rows),output=str(output))
