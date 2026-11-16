import json
import sqlite3

import numpy as np
import pytest
from click.testing import CliRunner
from chemsmart_ml.cli import build_cli
from chemsmart_ml.fingerprints import morgan_config, cached_morgan
from chemsmart_ml.workflow import train, predict
from test_workflow import database

pytest.importorskip('rdkit')


def test_cache_and_parameter_changes(tmp_path):
    rows=[dict(record_id='a',smiles='CCO'),dict(record_id='b',smiles='CCO')]
    cfg=morgan_config(size=64)
    cache=tmp_path/'cache.sqlite'
    a, stats=cached_morgan(rows,cfg,cache)
    assert stats['generated']==1 and stats['reused_in_batch']==1
    b, stats=cached_morgan(rows,cfg,cache)
    np.testing.assert_array_equal(a,b)
    assert stats['generated']==0 and stats['cache_hits']==1
    _, stats=cached_morgan(rows,morgan_config(radius=3,size=64),cache)
    assert stats['generated']==1
    with pytest.raises(ValueError,match='Invalid SMILES'):
        cached_morgan([dict(record_id='bad',smiles='invalid!')],cfg,cache)
    bad=dict(cfg,rdkit_version='wrong')
    with pytest.raises(ValueError,match='version'):
        cached_morgan(rows,bad,cache)


def add_smiles(path):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE molecules(molecule_id TEXT PRIMARY KEY,smiles TEXT)')
        db.executemany('INSERT INTO molecules VALUES (?,?)',[(f'm{i}','C') for i in range(120)])


def test_train_predict_and_prepare(tmp_path,monkeypatch):
    monkeypatch.setenv('CHEMSMART_CONFIG_DIR',str(tmp_path/'config'))
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True);add_smiles(a)
    cfg=tmp_path/'ml.yaml'
    cfg.write_text(f'data:\n  filename: {a}\n  reference: {b}\ntrain:\n  pairing: revqm9-index\n  fingerprint: morgan\n  morgan_size: 32\n  output: train\nprepare:\n  morgan_size: 32\n  output: prepared\n')
    from chemsmart_ml.cli import execute
    cli=build_cli(execute)
    result=CliRunner().invoke(cli,['-p',str(cfg),'prepare'])
    assert result.exit_code==0,result.output
    report=json.loads((tmp_path/'prepared/fingerprints.json').read_text())
    assert report['generated']==1
    result=CliRunner().invoke(cli,['-p',str(cfg),'train','--model','ridge'])
    assert result.exit_code==0,result.output
    report=json.loads((tmp_path/'train/fingerprints.json').read_text())
    assert report['generated']==0
    model=json.loads((tmp_path/'train/model.json').read_text())
    assert model['fingerprint']['size']==32
    assert len(model['mean'])==35
    predict(a,tmp_path/'train/model.json',tmp_path/'pred.csv')
    assert len((tmp_path/'pred.csv').read_text().splitlines())==121


def test_matches_chemsmart_rdkit_primitive(tmp_path):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    mol=Chem.MolFromSmiles('CC(=O)O')
    expected=rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2,nBits=128)
    bits=np.zeros(128,dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(expected,bits)
    actual,_=cached_morgan([dict(record_id='acid',smiles='CC(=O)O')],morgan_config(size=128),tmp_path/'cache.sqlite')
    np.testing.assert_array_equal(bits,actual[0])
