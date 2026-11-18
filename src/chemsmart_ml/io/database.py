"""
Read-only, indexed adapters for CHEMSMART SQLite databases.

This module contains:
- MolecularRecord: Final stored structure and properties of one record
- CHEMSMARTDatabase: Abstract reader selecting one structure per record
- SourceDatabase: Low-level (xTB) database indexed by ``idxN`` paths
- ReferenceDatabase: Reference (DFT) database indexed by NPZ position

Both concrete readers implement the explicit revQM9 convention: the
source path contains ``idxN`` (one-based) and the reference
``record_structures.structure_index_in_file`` is N (one-based). Both
databases MUST derive from the same NPZ ordering.
"""

import json
import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MolecularRecord:
    """
    Final stored structure and properties of one calculation record.

    Attributes:
        record_id (str): Database record identifier.
        energy (float): Total energy in Hartree.
        homo (float): HOMO energy in eV.
        lumo (float): LUMO energy in eV.
        dipole (float): Dipole moment magnitude in Debye.
        molecule_id (str): Stored molecular identity.
        atoms (list[str]): Chemical symbols in structure order.
        positions (list): Cartesian coordinates in Angstrom, if read.
        smiles (str): Final-structure SMILES, if read.
    """

    record_id: str
    energy: Optional[float]
    homo: Optional[float]
    lumo: Optional[float]
    dipole: Optional[float]
    molecule_id: Optional[str]
    atoms: List[str]
    positions: Optional[list] = None
    smiles: Optional[str] = None

    def has_finite(self, name: str) -> bool:
        """Return True if the named numeric property is present and finite."""
        value = getattr(self, name)
        return value is not None and bool(np.isfinite(value))

    def has_same_composition(self, other: "MolecularRecord") -> bool:
        """Return True if both records contain the same elements."""
        return sorted(self.atoms) == sorted(other.atoms)


class CHEMSMARTDatabase(ABC):
    """
    Abstract read-only reader for a CHEMSMART database.

    Only the final stored structure of each record (maximum
    ``index_in_record``) is read; trajectories are never expanded.
    Subclasses define how a record maps to its zero-based sample index.

    Attributes:
        filepath (Path): Resolved path to the SQLite database.
        include_smiles (bool): Whether to read final-structure SMILES.
        include_positions (bool): Whether to read Cartesian coordinates.
    """

    ROLE: Optional[str] = None

    def __init__(
        self,
        filepath,
        include_smiles: bool = False,
        include_positions: bool = False,
    ):
        """
        Initialize the database reader.

        Args:
            filepath (str | Path): Path to the CHEMSMART SQLite database.
            include_smiles (bool): Read SMILES from the molecules table.
            include_positions (bool): Read structure coordinates.
        """
        self.filepath = Path(filepath).resolve()
        self.include_smiles = include_smiles
        self.include_positions = include_positions

    def __repr__(self):
        return f"{self.__class__.__qualname__}<filepath={self.filepath}>"

    @property
    def identity(self) -> dict:
        """File identity used to detect changes since validation."""
        stat = self.filepath.stat()
        return dict(
            path=str(self.filepath),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    @cached_property
    def records(self) -> Dict[int, MolecularRecord]:
        """Final-structure records keyed by zero-based sample index."""
        uri = self.filepath.as_uri() + "?mode=ro"
        records = {}
        with closing(sqlite3.connect(uri, uri=True)) as db:
            for row in db.execute(self._query):
                key = self._sample_index(row)
                if key in records:
                    raise ValueError(
                        f"Duplicate sample index {key} in {self.filepath}"
                    )
                records[key] = self._record_from_row(row)
        if not records:
            raise ValueError(f"No samples in {self.filepath}")
        logger.debug(f"Read {len(records)} records from {self.filepath}")
        return dict(sorted(records.items()))

    @property
    def _query(self) -> str:
        """SQL selecting the final structure of every record."""
        columns = [
            "r.record_id",
            "r.source",
            "r.total_energy",
            "r.homo_energy",
            "r.lumo_energy",
            "s.molecule_id",
            "s.chemical_symbols_json",
            "rs.structure_index_in_file",
            "rs.dipole_moment_magnitude",
        ]
        joins = [
            "JOIN record_structures rs ON rs.record_id=r.record_id",
            "JOIN structures s ON s.structure_id=rs.structure_id",
        ]
        if self.include_positions:
            columns.append("s.positions_json")
        if self.include_smiles:
            columns.append("m.smiles")
            joins.append(
                "LEFT JOIN molecules m ON m.molecule_id=s.molecule_id"
            )
        return (
            f"SELECT {', '.join(columns)} FROM records r {' '.join(joins)} "
            "WHERE rs.index_in_record=(SELECT MAX(q.index_in_record) "
            "FROM record_structures q WHERE q.record_id=r.record_id)"
        )

    def _record_from_row(self, row) -> MolecularRecord:
        """Build a record from one row returned by ``_query``."""
        record_id, _, energy, homo, lumo, molecule_id, symbols = row[:7]
        dipole = row[8]
        extra = list(row[9:])
        positions = None
        smiles = None
        if self.include_positions:
            positions_json = extra.pop(0)
            positions = json.loads(positions_json) if positions_json else None
        if self.include_smiles:
            smiles = extra.pop(0)
        return MolecularRecord(
            record_id=record_id,
            energy=energy,
            homo=homo,
            lumo=lumo,
            dipole=dipole,
            molecule_id=molecule_id,
            atoms=json.loads(symbols),
            positions=positions,
            smiles=smiles,
        )

    @abstractmethod
    def _sample_index(self, row) -> int:
        """Return the zero-based dataset index of one query row."""


class SourceDatabase(CHEMSMARTDatabase):
    """Low-level database whose source paths contain one-based ``idxN``."""

    ROLE = "source"
    SOURCE_INDEX_PATTERN = re.compile(r"idx(\d+)(?:\D|$)")

    def _sample_index(self, row) -> int:
        source = row[1]
        match = self.SOURCE_INDEX_PATTERN.search(source or "")
        if not match:
            raise ValueError(f"Cannot extract source idx from {source!r}")
        return int(match.group(1)) - 1


class ReferenceDatabase(CHEMSMARTDatabase):
    """Reference database storing one-based NPZ structure positions."""

    ROLE = "reference"

    def _sample_index(self, row) -> int:
        record_id, index = row[0], row[7]
        if index is None or index < 1:
            raise ValueError(
                f"Missing one-based NPZ structure position: {record_id}"
            )
        return int(index) - 1
