"""Morgan bit fingerprints, sharing CHEMSMART's RDKit backend, with durable cache."""
import hashlib
import json
import sqlite3
from pathlib import Path

import click
import numpy as np

from .projects import project_directory


def morgan_config(radius=2, size=2048, chirality=False):
    try:
        from rdkit import rdBase
    except ImportError as exc:
        raise ValueError('Morgan requires RDKit; install chemsmart-ml[morgan]') from exc
    if not isinstance(radius, int) or radius < 0 or not isinstance(size, int) or size < 1:
        raise ValueError('Morgan radius must be nonnegative and size must be positive')
    return dict(method='morgan', radius=radius, size=size, chirality=chirality,
                representation='bits', remove_hydrogens=True, rdkit_version=rdBase.rdkitVersion)


def cached_morgan(rows, config, cache=None):
    """Read final-structure SMILES; never infer bonds from another trajectory frame."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    expected = morgan_config(config['radius'], config['size'], config['chirality'])
    if config != expected:
        raise ValueError('Fingerprint settings or RDKit version differ from the saved model')
    path = Path(cache) if cache else project_directory()/'cache'/'morgan.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(config, sort_keys=True)
    stats = dict(samples=len(rows), generated=0, cache_hits=0, reused_in_batch=0,
                 cache=str(path.resolve()), settings=config)
    result = np.empty((len(rows), config['size']), dtype=np.uint8)
    seen = {}
    generator = GetMorganGenerator(radius=config["radius"],fpSize=config["size"],includeChirality=config["chirality"])
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS fingerprints (key TEXT PRIMARY KEY, bits BLOB NOT NULL)')
        for i, row in enumerate(rows):
            smiles = row.get('smiles')
            if not smiles:
                raise ValueError(f"Missing final-structure SMILES for record {row['record_id']}")
            if smiles in seen:
                result[i] = result[seen[smiles]]
                stats['reused_in_batch'] += 1
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                db.commit()
                raise ValueError(f"Invalid SMILES for record {row['record_id']}: {smiles}")
            molecule = Chem.RemoveHs(molecule)
            canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
            key = hashlib.sha256((signature+'\n'+canonical).encode()).hexdigest()
            hit = db.execute('SELECT bits FROM fingerprints WHERE key=?', (key,)).fetchone()
            if hit:
                vector = np.unpackbits(np.frombuffer(hit[0],dtype=np.uint8))[:config['size']]
                if len(vector) != config['size']:
                    raise ValueError('Corrupt Morgan cache entry')
                stats['cache_hits'] += 1
            else:
                # Same RDKit primitive as CHEMSMART TanimotoGrouper; no pairwise grouping.
                fingerprint = generator.GetFingerprint(molecule)
                vector = np.zeros(config['size'], dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fingerprint, vector)
                db.execute('INSERT OR IGNORE INTO fingerprints VALUES (?,?)',
                           (key, np.packbits(vector).tobytes()))
                stats['generated'] += 1
            result[i] = vector
            seen[smiles] = i
            if (i+1)%1000 == 0:
                db.commit()
                click.echo(f'Morgan: {i+1}/{len(rows)} samples', err=True)
    return result, stats
