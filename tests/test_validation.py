import csv
import json
import sqlite3
import pytest
from test_workflow import database
from test_fingerprints import add_smiles
from chemsmart_ml.validation import validate,load_manifest
from chemsmart_ml.workflow import prepare,train


def test_full_screen_manifest_prepare_train(tmp_path,monkeypatch):
    monkeypatch.setenv('CHEMSMART_CONFIG_DIR',str(tmp_path/'config'))
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True);add_smiles(a)
    with sqlite3.connect(a) as db:
        db.execute("UPDATE molecules SET smiles='invalid!' WHERE molecule_id='m3'")
        db.execute("UPDATE records SET homo_energy=NULL WHERE record_id='4'")
    report=validate(a,b,tmp_path/'validation')
    assert report['valid']==118 and report['excluded']==2
    manifest=tmp_path/'validation/valid-samples.json'
    excluded=list(csv.DictReader((tmp_path/'validation/excluded.csv').open()))
    assert {int(row['idx']) for row in excluded}=={4,5}
    prepare(a,tmp_path/'prepared',morgan_size=32,manifest=manifest)
    train(a,b,tmp_path/'trained',electronic='orbitals',fingerprint='morgan',morgan_size=32,model='ridge',manifest=manifest)
    rows=list(csv.DictReader((tmp_path/'trained/samples.csv').open()))
    assert len(rows)==118 and not {'3','4'} & {r['npz_index'] for r in rows}
    # Same filtered sample set can be used for a fair numerical baseline.
    train(a,b,tmp_path/'baseline',electronic='orbitals',manifest=manifest)
    base=list(csv.DictReader((tmp_path/'baseline/samples.csv').open()))
    assert [(r['npz_index'],r['split']) for r in base]==[(r['npz_index'],r['split']) for r in rows]
    with sqlite3.connect(a) as db:
        db.execute("UPDATE records SET homo_energy=0 WHERE record_id='4'")
    with pytest.raises(ValueError,match='source database'):
        load_manifest(manifest,a,b)
