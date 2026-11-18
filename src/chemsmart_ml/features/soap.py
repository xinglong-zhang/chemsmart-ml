"""SOAP descriptors from CHEMSMART's molecular geometry implementation."""

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from chemsmart_ml.features.base import CachedDescriptor

logger = logging.getLogger(__name__)


class SOAPDescriptor(CachedDescriptor):
    """
    Smooth Overlap of Atomic Positions descriptor of a final geometry.

    Settings record the CHEMSMART version, a hash of the SOAP
    implementation source and NumPy/SciPy versions, so a saved model is
    rejected if the numerical implementation changes. Parameters are
    fixed at construction.

    Attributes:
        species (list[str]): Element basis; inputs may not exceed it.
        r_cut (float): Cutoff radius in Angstrom (> 1).
        n_max (int): Number of radial basis functions.
        l_max (int): Maximum spherical-harmonic degree.
        sigma (float): Gaussian width in Angstrom.
        aggregation (str): Per-atom aggregation, 'sum' or 'mean'.
        cache (Path): SQLite cache file.
    """

    NAME = "SOAP"
    CACHE_FILENAME = "soap.sqlite"
    TABLE = "descriptors"
    VALUE_COLUMN = "vector"
    DTYPE = float
    COMMIT_INTERVAL = 100
    REPORT_FILENAME = "soap.json"
    ERROR_FILENAME = "soap-error.json"
    AGGREGATIONS = ("sum", "mean")

    def __init__(
        self,
        species: Sequence[str],
        r_cut: float = 5.0,
        n_max: int = 2,
        l_max: int = 2,
        sigma: float = 0.5,
        aggregation: str = "sum",
        cache=None,
    ):
        """
        Initialize a SOAP descriptor.

        Args:
            species: Element basis, normally the training elements.
            r_cut (float): Cutoff radius in Angstrom; must exceed 1.
            n_max (int): Number of radial basis functions.
            l_max (int): Maximum spherical-harmonic degree.
            sigma (float): Gaussian width in Angstrom.
            aggregation (str): 'sum' or 'mean' over atomic centres.
            cache (str | Path, optional): SQLite cache file.
        """
        super().__init__(cache)
        try:
            import chemsmart
            import scipy
            from chemsmart.analysis.soap import calculate_soap
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
            or aggregation not in self.AGGREGATIONS
        ):
            raise ValueError("Invalid SOAP parameters")
        self._species = list(species)
        self._r_cut = r_cut
        self._n_max = n_max
        self._l_max = l_max
        self._sigma = sigma
        self._aggregation = aggregation
        source = Path(inspect.getfile(calculate_soap)).read_bytes()
        self._versions = dict(
            chemsmart_version=chemsmart.__version__,
            implementation_sha256=hashlib.sha256(source).hexdigest(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
        )

    @classmethod
    def from_settings(cls, settings: dict, cache=None) -> "SOAPDescriptor":
        descriptor = cls(
            settings["species"],
            settings["r_cut"],
            settings["n_max"],
            settings["l_max"],
            settings["sigma"],
            settings["aggregation"],
            cache,
        )
        if descriptor.settings != settings:
            raise ValueError(
                "SOAP implementation or dependency versions differ from "
                "saved model"
            )
        return descriptor

    @property
    def species(self):
        return list(self._species)

    @property
    def r_cut(self):
        return self._r_cut

    @property
    def n_max(self):
        return self._n_max

    @property
    def l_max(self):
        return self._l_max

    @property
    def sigma(self):
        return self._sigma

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def settings(self) -> dict:
        return dict(
            method="soap",
            species=self.species,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            aggregation=self.aggregation,
            **self._versions,
        )

    def _prepare(self, record):
        positions = np.asarray(record.positions, dtype=float)
        if (
            positions.shape != (len(record.atoms), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError(f"Invalid geometry for record {record.record_id}")
        if set(record.atoms) - set(self._species):
            raise ValueError(
                "SOAP species not present in training basis for record "
                f"{record.record_id}"
            )
        signature = json.dumps(
            self.settings, sort_keys=True, separators=(",", ":")
        )
        key = hashlib.sha256(
            signature.encode()
            + json.dumps(record.atoms, separators=(",", ":")).encode()
            + positions.astype("<f8").tobytes()
        ).hexdigest()
        return key, positions

    def _calculate(self, record, positions) -> np.ndarray:
        from chemsmart.io.molecules.structure import Molecule

        molecule = Molecule(symbols=record.atoms, positions=positions)
        vector = np.asarray(
            molecule.calculate_soap(
                r_cut=self.r_cut,
                n_max=self.n_max,
                l_max=self.l_max,
                sigma=self.sigma,
                species=self.species,
                aggregation=self.aggregation,
            ),
            dtype="<f8",
        ).reshape(-1)
        if not np.isfinite(vector).all() or not vector.size:
            raise ValueError(
                f"Invalid SOAP vector for record {record.record_id}"
            )
        return vector

    def _encode(self, vector) -> bytes:
        return np.asarray(vector, dtype="<f8").tobytes()

    def _decode(self, blob) -> np.ndarray:
        return np.frombuffer(blob, dtype="<f8").copy()
