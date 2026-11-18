"""Morgan bit fingerprints sharing CHEMSMART's RDKit backend."""

import hashlib
import json
import logging

import numpy as np

from chemsmart_ml.features.base import CachedDescriptor

logger = logging.getLogger(__name__)


class MorganFingerprint(CachedDescriptor):
    """
    Morgan bit fingerprint of the final-structure SMILES.

    Bonds are taken from the stored SMILES and are never inferred from
    another trajectory frame. Hydrogens are removed before hashing.
    Fingerprint parameters are fixed at construction.

    Attributes:
        radius (int): Morgan radius.
        size (int): Number of bits.
        chirality (bool): Whether chirality is included.
        cache (Path): SQLite cache file.
    """

    NAME = "Morgan"
    CACHE_FILENAME = "morgan.sqlite"
    TABLE = "fingerprints"
    VALUE_COLUMN = "bits"
    DTYPE = np.uint8
    COMMIT_INTERVAL = 1000
    REPORT_FILENAME = "fingerprints.json"
    ERROR_FILENAME = "fingerprint-error.json"
    DEDUPLICATE_BATCH = True

    def __init__(
        self,
        radius: int = 2,
        size: int = 2048,
        chirality: bool = False,
        cache=None,
    ):
        """
        Initialize a Morgan fingerprint.

        Args:
            radius (int): Morgan radius; must be nonnegative.
            size (int): Number of fingerprint bits; must be positive.
            chirality (bool): Include chirality in atom invariants.
            cache (str | Path, optional): SQLite cache file.
        """
        super().__init__(cache)
        try:
            from rdkit import rdBase
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        except ImportError as exc:
            raise ValueError(
                "Morgan requires RDKit; install chemsmart-ml[morgan]"
            ) from exc
        if (
            not isinstance(radius, int)
            or radius < 0
            or not isinstance(size, int)
            or size < 1
        ):
            raise ValueError(
                "Morgan radius must be nonnegative and size must be positive"
            )
        self._radius = radius
        self._size = size
        self._chirality = chirality
        self._rdkit_version = rdBase.rdkitVersion
        # Same RDKit primitive as CHEMSMART TanimotoGrouper.
        self._generator = GetMorganGenerator(
            radius=radius, fpSize=size, includeChirality=chirality
        )

    @classmethod
    def from_settings(cls, settings: dict, cache=None) -> "MorganFingerprint":
        fingerprint = cls(
            settings["radius"], settings["size"], settings["chirality"], cache
        )
        if fingerprint.settings != settings:
            raise ValueError(
                "Fingerprint settings or RDKit version differ from the "
                "saved model"
            )
        return fingerprint

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def size(self) -> int:
        return self._size

    @property
    def chirality(self) -> bool:
        return self._chirality

    @property
    def settings(self) -> dict:
        return dict(
            method="morgan",
            radius=self.radius,
            size=self.size,
            chirality=self.chirality,
            representation="bits",
            remove_hydrogens=True,
            rdkit_version=self._rdkit_version,
        )

    @staticmethod
    def is_valid_smiles(smiles) -> bool:
        """Return True if RDKit parses the SMILES, without logging noise."""
        from rdkit import Chem, rdBase

        if not smiles:
            return False
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles) is not None

    def _batch_key(self, record) -> str:
        if not record.smiles:
            raise ValueError(
                "Missing final-structure SMILES for record "
                f"{record.record_id}"
            )
        return record.smiles

    def _prepare(self, record):
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(record.smiles)
        if molecule is None:
            raise ValueError(
                f"Invalid SMILES for record {record.record_id}: "
                f"{record.smiles}"
            )
        molecule = Chem.RemoveHs(molecule)
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        signature = json.dumps(self.settings, sort_keys=True)
        key = hashlib.sha256(
            (signature + "\n" + canonical).encode()
        ).hexdigest()
        return key, molecule

    def _calculate(self, record, molecule) -> np.ndarray:
        from rdkit import DataStructs

        fingerprint = self._generator.GetFingerprint(molecule)
        vector = np.zeros(self.size, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        return vector

    def _encode(self, vector) -> bytes:
        return np.packbits(vector).tobytes()

    def _decode(self, blob) -> np.ndarray:
        vector = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
        vector = vector[: self.size]
        if len(vector) != self.size:
            raise ValueError("Corrupt Morgan cache entry")
        return vector
