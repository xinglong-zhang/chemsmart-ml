"""Base class for per-structure descriptors with a durable SQLite cache."""

import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from chemsmart_ml.io.database import MolecularRecord
from chemsmart_ml.projects import project_directory

logger = logging.getLogger(__name__)


class CachedDescriptor(ABC):
    """
    Abstract descriptor computed per record and cached across runs.

    Cache keys hash the complete descriptor settings with the structure
    input, so changing parameters or dependency versions never reuses a
    stale vector. Subclasses define keys, calculation and encoding.

    Attributes:
        cache (Path): SQLite cache file.
    """

    NAME: Optional[str] = None
    CACHE_FILENAME: Optional[str] = None
    TABLE: Optional[str] = None
    VALUE_COLUMN: Optional[str] = None
    DTYPE = float
    COMMIT_INTERVAL = 1000
    REPORT_FILENAME: Optional[str] = None
    ERROR_FILENAME: Optional[str] = None
    # Reuse vectors for records with an identical batch key in one call.
    DEDUPLICATE_BATCH = False

    def __init__(self, cache=None):
        """
        Initialize the descriptor cache location.

        Args:
            cache (str | Path, optional): SQLite cache file. Defaults to
                ``cache/<CACHE_FILENAME>`` under the ML project directory.
        """
        if cache:
            self.cache = Path(cache)
        else:
            self.cache = project_directory() / "cache" / self.CACHE_FILENAME

    def __repr__(self):
        return f"{self.__class__.__qualname__}<{self.settings}>"

    @property
    @abstractmethod
    def settings(self) -> dict:
        """JSON-serialisable settings that fully determine the vectors."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: dict, cache=None) -> "CachedDescriptor":
        """Rebuild a descriptor from saved settings, rejecting mismatches."""

    def compute(
        self, records: Sequence[MolecularRecord]
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute or load one descriptor vector per record.

        Returns:
            tuple: (matrix with one row per record, cache report).
        """
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        counts = dict(generated=0, cache_hits=0, reused_in_batch=0)
        vectors: List[np.ndarray] = []
        seen = {}
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE} "
                f"(key TEXT PRIMARY KEY, {self.VALUE_COLUMN} BLOB NOT NULL)"
            )
            try:
                for i, record in enumerate(records):
                    batch_key = None
                    if self.DEDUPLICATE_BATCH:
                        batch_key = self._batch_key(record)
                        if batch_key in seen:
                            vectors.append(vectors[seen[batch_key]])
                            counts["reused_in_batch"] += 1
                            continue
                    vector = self._lookup_or_calculate(db, record, counts)
                    if vectors and len(vector) != len(vectors[0]):
                        raise ValueError(
                            f"Inconsistent {self.NAME} descriptor dimensions"
                        )
                    vectors.append(vector)
                    if batch_key is not None:
                        seen[batch_key] = i
                    if (i + 1) % self.COMMIT_INTERVAL == 0:
                        db.commit()
                        logger.info(
                            f"{self.NAME}: {i + 1}/{len(records)} samples"
                        )
            except Exception:
                # Keep vectors already generated before the failure.
                db.commit()
                raise
        report = dict(samples=len(records), **self._report_counts(counts))
        report.update(cache=str(self.cache.resolve()), settings=self.settings)
        return np.asarray(vectors, dtype=self.DTYPE), report

    def _lookup_or_calculate(self, db, record, counts) -> np.ndarray:
        key, prepared = self._prepare(record)
        hit = db.execute(
            f"SELECT {self.VALUE_COLUMN} FROM {self.TABLE} WHERE key=?",
            (key,),
        ).fetchone()
        if hit:
            counts["cache_hits"] += 1
            return self._decode(hit[0])
        vector = self._calculate(record, prepared)
        db.execute(
            f"INSERT OR IGNORE INTO {self.TABLE} VALUES (?,?)",
            (key, self._encode(vector)),
        )
        counts["generated"] += 1
        return vector

    def _report_counts(self, counts: dict) -> dict:
        if not self.DEDUPLICATE_BATCH:
            counts = {
                k: v for k, v in counts.items() if k != "reused_in_batch"
            }
        return counts

    def _batch_key(self, record: MolecularRecord):
        """Key identifying records whose vectors are identical in a batch."""
        raise NotImplementedError

    @abstractmethod
    def _prepare(self, record: MolecularRecord) -> Tuple[str, object]:
        """Validate a record; return its cache key and calculation input."""

    @abstractmethod
    def _calculate(self, record: MolecularRecord, prepared) -> np.ndarray:
        """Calculate the descriptor vector for a cache miss."""

    @abstractmethod
    def _encode(self, vector: np.ndarray) -> bytes:
        """Serialise a vector for the cache."""

    @abstractmethod
    def _decode(self, blob: bytes) -> np.ndarray:
        """Deserialise a cached vector."""
