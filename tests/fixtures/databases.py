"""Fixtures providing small paired CHEMSMART databases."""

import pytest

from tests.helpers import add_geometry, add_smiles, create_database


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Isolated CHEMSMART configuration root for caches and projects."""
    path = tmp_path / "config"
    monkeypatch.setenv("CHEMSMART_CONFIG_DIR", str(path))
    return path


@pytest.fixture
def paired_databases(tmp_path):
    """Source and reference databases of 120 paired C/H samples."""
    source, reference = tmp_path / "source.db", tmp_path / "reference.db"
    create_database(source)
    create_database(reference, reference=True)
    return source, reference


@pytest.fixture
def smiles_databases(paired_databases):
    """Paired databases whose source has final-structure SMILES."""
    add_smiles(paired_databases[0])
    return paired_databases


@pytest.fixture
def geometry_databases(smiles_databases):
    """Paired databases whose source has SMILES and coordinates."""
    add_geometry(smiles_databases[0])
    return smiles_databases
