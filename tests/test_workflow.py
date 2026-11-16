import json
import sqlite3
import numpy as np
import pytest
from chemsmart_ml.data import read_database, pair
from chemsmart_ml.workflow import train, predict, groups


def database(path, reference=False):
    db=sqlite3.connect(path)
    db.executescript('''
    CREATE TABLE records(record_id TEXT,source TEXT,total_energy REAL,homo_energy REAL,lumo_energy REAL);
    CREATE TABLE structures(structure_id TEXT,molecule_id TEXT,chemical_symbols_json TEXT);
    CREATE TABLE record_structures(record_id TEXT,structure_id TEXT,index_in_record INTEGER,structure_index_in_file INTEGER,dipole_moment_magnitude REAL);
    CREATE TABLE dataset_properties(record_id TEXT,properties_json TEXT);
    ''')
    for i in range(120):
        energy=-10-i*.1
        db.execute('INSERT INTO records VALUES(?,?,?,?,?)',(str(i),f'/opt_idx{i+1}',energy*1.2+3 if reference else energy,-1,1))
        db.execute('INSERT INTO structures VALUES(?,?,?)',(str(i),f'm{i}',json.dumps(['C','H'])))
        db.execute('INSERT INTO record_structures VALUES(?,?,?,?,?)',(str(i),str(i),1,i+1,0.5))
        db.execute('INSERT INTO dataset_properties VALUES(?,?)',(str(i),json.dumps({'inds':{'value':i+1000}})))
    db.commit();db.close()


def test_pair_train_predict(tmp_path):
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    source,ref=read_database(a,'source'),read_database(b,'reference')
    assert set(source)==set(range(120))
    assert pair(source,ref)[1]['matched']==120
    metrics=train(a,b,tmp_path/'run')
    assert metrics['test']['mae_hartree']<1e-10
    predict(a,tmp_path/'run/model.json',tmp_path/'pred.csv')
    assert len((tmp_path/'pred.csv').read_text().splitlines())==121
    with pytest.raises(FileExistsError):
        train(a,b,tmp_path/'run')


def test_group_transitivity():
    p=[(0,{'molecule_id':'A'},{'molecule_id':'B'}),
       (1,{'molecule_id':'C'},{'molecule_id':'B'})]
    assert len(set(groups(p)))==1


def test_duplicate_and_composition(tmp_path):
    a=tmp_path/'a.db';database(a)
    with sqlite3.connect(a) as db:
        db.execute("UPDATE records SET source='/opt_idx1' WHERE record_id='1'")
    with pytest.raises(ValueError,match='Duplicate'):
        read_database(a,'source')
    pairs,report=pair({0:{'atoms':['C']}},{0:{'atoms':['N']}})
    assert not pairs and len(report['rejected'])==1


def test_ridge_roundtrip(tmp_path):
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    train(a,b,tmp_path/'ridge',model='ridge',alpha=0.1)
    model=json.loads((tmp_path/'ridge/model.json').read_text())
    assert model['model']=='ridge-delta'
    predict(a,tmp_path/'ridge/model.json',tmp_path/'ridge.csv')
    assert len((tmp_path/'ridge.csv').read_text().splitlines())==121
