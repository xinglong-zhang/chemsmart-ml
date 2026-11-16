"""Read-only, indexed CHEMSMART database adapters."""
import json
import re
import sqlite3
from pathlib import Path


def read_database(path, role, include_smiles=False):
    """Explicit revQM9 convention: source idx is one-based; NPZ structure_index_in_file is one-based."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        # Use only the last structure per calculation. Never expand trajectories.
        query = """
        SELECT r.record_id,r.source,r.total_energy,r.homo_energy,r.lumo_energy,
               s.molecule_id,s.chemical_symbols_json,rs.structure_index_in_file,
               rs.dipole_moment_magnitude
        FROM records r JOIN record_structures rs ON rs.record_id=r.record_id
        JOIN structures s ON s.structure_id=rs.structure_id
        WHERE rs.index_in_record=(SELECT MAX(q.index_in_record)
            FROM record_structures q WHERE q.record_id=r.record_id)
        """
        if include_smiles:
            query = query.replace('rs.dipole_moment_magnitude\n', 'rs.dipole_moment_magnitude, m.smiles\n')
            query = query.replace('WHERE rs.index_in_record=', 'LEFT JOIN molecules m ON m.molecule_id=s.molecule_id WHERE rs.index_in_record=')
        rows = {}
        for record in db.execute(query):
            rid, source, energy, homo, lumo, mid, symbols, index, dipole = record[:9]
            if role == "source":
                match = re.search(r"idx(\d+)(?:\D|$)", source or "")
                if not match:
                    raise ValueError(f"Cannot extract source idx from {source!r}")
                key = int(match.group(1)) - 1
            else:
                if index is None or index < 1:
                    raise ValueError(f"Missing one-based NPZ structure position: {rid}")
                key = int(index) - 1
            if key in rows:
                raise ValueError(f"Duplicate sample index {key} in {path}")
            atoms = json.loads(symbols)
            rows[key] = dict(record_id=rid, energy=energy, homo=homo, lumo=lumo,
                             dipole=dipole, molecule_id=mid, atoms=atoms)
            if include_smiles:
                rows[key]["smiles"] = record[9]
        if not rows:
            raise ValueError(f"No samples in {path}")
        return rows


def pair(source, reference):
    matched, rejected = [], []
    for key in sorted(source.keys() & reference.keys()):
        a, b = source[key], reference[key]
        if sorted(a["atoms"]) != sorted(b["atoms"]):
            rejected.append(dict(index=key, reason="element composition mismatch"))
            continue
        matched.append((key, a, b))
    return matched, dict(source_only=sorted(source.keys()-reference.keys()),
                         reference_only=sorted(reference.keys()-source.keys()),
                         rejected=rejected, matched=len(matched),
                         identity_changes=sum(a["molecule_id"] != b["molecule_id"] for _,a,b in matched))
