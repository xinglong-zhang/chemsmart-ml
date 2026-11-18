import numpy as np
import pytest

from chemsmart_ml.features import FeatureBuilder, FeatureScaler
from tests.helpers import make_record


def test_orbitals_feature_set_does_not_require_dipole():
    row = make_record("sample", energy=-1, homo=-2, lumo=1)
    assert FeatureBuilder(["C"], "orbitals").base_vector(row) == [1, -1, -2, 1]
    with pytest.raises(ValueError, match="dipole"):
        FeatureBuilder(["C"], "electronic").base_vector(row)
    with pytest.raises(ValueError, match="absent from training"):
        FeatureBuilder(["H"]).base_vector(row)


def test_electronic_artifact_values_round_trip():
    for name in FeatureBuilder.ELECTRONIC_KEYS:
        electronic = FeatureBuilder(["C"], name).electronic
        assert FeatureBuilder.features_from_electronic(electronic) == name


def test_scaler_keeps_constant_columns_finite():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    design = FeatureScaler().fit(x).design_matrix(x)
    np.testing.assert_array_equal(design, [[1, -1, 0], [1, 1, 0]])


def test_morgan_cache_and_parameter_changes(tmp_path):
    pytest.importorskip("rdkit")
    from chemsmart_ml.features import MorganFingerprint

    rows = [make_record("a", smiles="CCO"), make_record("b", smiles="CCO")]
    cache = tmp_path / "cache.sqlite"
    fingerprint = MorganFingerprint(size=64, cache=cache)
    a, stats = fingerprint.compute(rows)
    assert stats["generated"] == 1 and stats["reused_in_batch"] == 1
    b, stats = fingerprint.compute(rows)
    np.testing.assert_array_equal(a, b)
    assert stats["generated"] == 0 and stats["cache_hits"] == 1
    _, stats = MorganFingerprint(radius=3, size=64, cache=cache).compute(rows)
    assert stats["generated"] == 1
    with pytest.raises(ValueError, match="Invalid SMILES"):
        fingerprint.compute([make_record("bad", smiles="invalid!")])
    bad = dict(fingerprint.settings, rdkit_version="wrong")
    with pytest.raises(ValueError, match="version"):
        MorganFingerprint.from_settings(bad, cache)


def test_morgan_matches_chemsmart_rdkit_primitive(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors

    from chemsmart_ml.features import MorganFingerprint

    mol = Chem.MolFromSmiles("CC(=O)O")
    expected = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=128
    )
    bits = np.zeros(128, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(expected, bits)
    fingerprint = MorganFingerprint(size=128, cache=tmp_path / "cache.sqlite")
    actual, _ = fingerprint.compute([make_record("acid", smiles="CC(=O)O")])
    np.testing.assert_array_equal(bits, actual[0])


def test_soap_cache_and_geometry_invariance(tmp_path):
    pytest.importorskip("chemsmart.analysis.soap")
    from chemsmart_ml.features import SOAPDescriptor

    row = make_record("test", ["C", "H"], positions=[[0, 0, 0], [1.1, 0, 0]])
    soap = SOAPDescriptor(["C", "H"], cache=tmp_path / "cache.db")
    first, report = soap.compute([row])
    assert report["generated"] == 1 and first.shape[1] > 0
    again, report = soap.compute([row])
    assert report["cache_hits"] == 1
    np.testing.assert_allclose(first, again)
    translated = make_record(
        "test", ["C", "H"], positions=[[5, 2, 1], [6.1, 2, 1]]
    )
    new, _ = soap.compute([translated])
    np.testing.assert_allclose(first, new, atol=1e-10)
    bad = make_record("test", ["C", "H"], positions=[[0, 0, 0]])
    with pytest.raises(ValueError, match="Invalid geometry"):
        soap.compute([bad])
    assert (
        SOAPDescriptor.from_settings(soap.settings).settings == soap.settings
    )
