"""Cached SOAP descriptors from the host's molecular geometry implementation."""

import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

import click
import numpy as np
import scipy

from .projects import project_directory


def soap_config(species, r_cut=5.0, n_max=2, l_max=2, sigma=0.5, aggregation="sum"):
    try:
        from chemsmart.analysis.soap import calculate_soap
        import chemsmart
    except ImportError as exc:
        raise ValueError(
            "SOAP requires the CHEMSMART plugin branch with SOAP support"
        ) from exc
    if (
        not np.isfinite([r_cut, sigma]).all()
        or r_cut <= 1
        or n_max < 1
        or l_max < 0
        or sigma <= 0
        or aggregation not in ("sum", "mean")
    ):
        raise ValueError("Invalid SOAP parameters")
    source = Path(inspect.getfile(calculate_soap)).read_bytes()
    return dict(
        method="soap",
        species=list(species),
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        aggregation=aggregation,
        chemsmart_version=chemsmart.__version__,
        implementation_sha256=hashlib.sha256(source).hexdigest(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )


def cached_soap(rows, config, cache=None):
    from chemsmart.io.molecules.structure import Molecule

    expected = soap_config(
        config["species"],
        config["r_cut"],
        config["n_max"],
        config["l_max"],
        config["sigma"],
        config["aggregation"],
    )
    if expected != config:
        raise ValueError(
            "SOAP implementation or dependency versions differ from saved model"
        )
    path = Path(cache) if cache else project_directory() / "cache" / "soap.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(config, sort_keys=True, separators=(",", ":"))
    vectors = []
    generated = 0
    hits = 0
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS descriptors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        for i, row in enumerate(rows):
            positions = np.asarray(row.get("positions"), dtype=float)
            if (
                positions.shape != (len(row["atoms"]), 3)
                or not np.isfinite(positions).all()
            ):
                raise ValueError(f"Invalid geometry for record {row['record_id']}")
            if set(row["atoms"]) - set(config["species"]):
                raise ValueError(
                    f"SOAP species not present in training basis for record {row['record_id']}"
                )
            key = hashlib.sha256(
                signature.encode()
                + json.dumps(row["atoms"], separators=(",", ":")).encode()
                + positions.astype("<f8").tobytes()
            ).hexdigest()
            hit = db.execute(
                "SELECT vector FROM descriptors WHERE key=?", (key,)
            ).fetchone()
            if hit:
                vec = np.frombuffer(hit[0], dtype="<f8").copy()
                hits += 1
            else:
                molecule = Molecule(symbols=row["atoms"], positions=positions)
                vec = np.asarray(
                    molecule.calculate_soap(
                        r_cut=config["r_cut"],
                        n_max=config["n_max"],
                        l_max=config["l_max"],
                        sigma=config["sigma"],
                        species=config["species"],
                        aggregation=config["aggregation"],
                    ),
                    dtype="<f8",
                ).reshape(-1)
                if not np.isfinite(vec).all() or not vec.size:
                    raise ValueError(
                        f"Invalid SOAP vector for record {row['record_id']}"
                    )
                db.execute(
                    "INSERT OR IGNORE INTO descriptors VALUES (?,?)",
                    (key, vec.tobytes()),
                )
                generated += 1
            if vectors and len(vec) != len(vectors[0]):
                raise ValueError("Inconsistent SOAP descriptor dimensions")
            vectors.append(vec)
            if (i + 1) % 100 == 0:
                db.commit()
                click.echo(f"SOAP: {i + 1}/{len(rows)} samples", err=True)
    return np.asarray(vectors), dict(
        samples=len(rows),
        generated=generated,
        cache_hits=hits,
        cache=str(path.resolve()),
        settings=config,
    )
