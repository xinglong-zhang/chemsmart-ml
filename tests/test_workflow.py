import json
from pathlib import Path
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


def test_yaml_precedence_and_paths(tmp_path):
    from click.testing import CliRunner
    from chemsmart_ml.cli import build_cli
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    project=tmp_path/'ml.yaml'
    project.write_text('data:\n  filename: a.db\n  reference: b.db\ntrain:\n  pairing: revqm9-index\n  model: ridge\n  alpha: 2.0\n  output: run\n')
    calls=[]
    cli=build_cli(lambda action,options:calls.append((action,options)))
    runner=CliRunner()
    result=runner.invoke(cli,['-p',str(project),'train','--model','linear','--alpha','0.5'])
    assert result.exit_code==0,result.output
    options=calls[-1][1]
    assert options['model']=='linear' and options['alpha']==0.5
    assert options['output']==str(tmp_path/'run')
    assert options['filename']==str(a)
    result=runner.invoke(cli,['-p',str(project),'train'])
    assert result.exit_code==0,result.output
    assert calls[-1][1]['model']=='ridge'
    project.write_text('train:\n  typo: 3\n')
    result=runner.invoke(cli,['-p',str(project),'train'])
    assert result.exit_code!=0 and 'Unknown keys' in result.output


def test_plot_cli_without_database(tmp_path):
    from click.testing import CliRunner
    from chemsmart_ml.cli import build_cli
    pytest.importorskip('matplotlib')
    (tmp_path/'samples.csv').write_text('split,reference_energy,prediction\ntest,-10,-9.99\ntrain,-20,-20\n')
    cli=build_cli(lambda *args:pytest.fail('Plot must not create a training job'))
    result=CliRunner().invoke(cli,['plot','--workdir',str(tmp_path)])
    assert result.exit_code==0,result.output
    assert (tmp_path/'plots/test-results.png').is_file()
    stats=json.loads((tmp_path/'plots/test-statistics.json').read_text())
    assert stats['n']==1
    assert stats['mae_kcal_mol']==pytest.approx(6.27509474)
    result=CliRunner().invoke(cli,['plot','--workdir',str(tmp_path)])
    assert result.exit_code!=0 and 'already exist' in result.output


def test_named_project_completion_and_override(tmp_path,monkeypatch):
    from click.testing import CliRunner
    from chemsmart_ml.cli import build_cli
    from chemsmart_ml.projects import complete_project
    monkeypatch.setenv('CHEMSMART_CONFIG_DIR',str(tmp_path))
    root=tmp_path/'ml';root.mkdir()
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    (root/'electronic.yaml').write_text('data:\n  filename: ../a.db\n  reference: ../b.db\ntrain:\n  pairing: revqm9-index\n  model: ridge\n  output: runs/test\n')
    calls=[]
    cli=build_cli(lambda action,options:calls.append(options))
    result=CliRunner().invoke(cli,['-p','electronic','train','--model','linear'])
    assert result.exit_code==0,result.output
    assert calls[0]['model']=='linear' and calls[0]['filename']==str(a)
    assert [item.value for item in complete_project(None,None,'ele')]==['electronic']
    result=CliRunner().invoke(cli,['-p','missing','train'])
    assert result.exit_code!=0 and 'not found' in result.output


def test_orbitals_without_dipole(tmp_path):
    from chemsmart_ml.workflow import features
    row=dict(atoms=['C'],energy=-1,homo=-2,lumo=1,dipole=None,record_id='sample')
    assert features(row,['C'],'orbitals')==[1,-1,-2,1]
    with pytest.raises(ValueError,match='dipole'):
        features(row,['C'],True)
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    with sqlite3.connect(a) as db:
        db.execute('UPDATE record_structures SET dipole_moment_magnitude=NULL')
    train(a,b,tmp_path/'orbitals',electronic='orbitals')
    predict(a,tmp_path/'orbitals/model.json',tmp_path/'orbitals.csv')
    counts=json.loads((tmp_path/'orbitals/feature-availability.json').read_text())
    assert counts['finite_counts']['dipole']==0


def test_train_automatic_plot_and_opt_out(tmp_path):
    from click.testing import CliRunner
    from chemsmart_ml.cli import build_cli,execute
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    train(a,b,tmp_path/'auto')
    assert (tmp_path/'auto/plots/test-results.png').is_file()
    assert (tmp_path/'auto/plots/test-results.pdf').is_file()
    stats=json.loads((tmp_path/'auto/plots/test-statistics.json').read_text())
    assert stats['n']==json.loads((tmp_path/'auto/metrics.json').read_text())['test']['n']
    cfg=tmp_path/'project.yaml'
    cfg.write_text(f'data:\n  filename: {a}\n  reference: {b}\ntrain:\n  pairing: revqm9-index\n  plot: false\n  output: disabled\n')
    cli=build_cli(execute)
    result=CliRunner().invoke(cli,['-p',str(cfg),'train'])
    assert result.exit_code==0,result.output
    assert not (tmp_path/'disabled/plots').exists()
    result=CliRunner().invoke(cli,['-p',str(cfg),'train','--plot','-o',str(tmp_path/'override')])
    assert result.exit_code==0,result.output
    assert (tmp_path/'override/plots/test-results.png').exists()
    result=CliRunner().invoke(cli,['-f',str(a),'-r',str(b),'train','--pairing','revqm9-index','--no-plot','-o',str(tmp_path/'no-plot')])
    assert result.exit_code==0,result.output
    assert not (tmp_path/'no-plot/plots').exists()


def test_default_output_is_invocation_directory(tmp_path,monkeypatch):
    from click.testing import CliRunner
    from chemsmart_ml.cli import build_cli
    a,b=tmp_path/'a.db',tmp_path/'b.db'
    database(a);database(b,True)
    presets=tmp_path/'presets';presets.mkdir()
    cfg=presets/'test.yaml'
    cfg.write_text('train:\n  pairing: revqm9-index\n')
    working=tmp_path/'working';working.mkdir()
    monkeypatch.chdir(working)
    calls=[]
    cli=build_cli(lambda action,options:calls.append(options))
    runner=CliRunner()
    args=['-p',str(cfg),'-f',str(a),'-r',str(b),'train']
    result=runner.invoke(cli,args)
    assert result.exit_code==0,result.output
    first=Path(calls[-1]['output'])
    assert first.parent==working and first.name.startswith('ml-train-')
    result=runner.invoke(cli,args)
    assert result.exit_code==0,result.output
    assert calls[-1]['output']!=str(first)
    result=runner.invoke(cli,args+['-o','chosen'])
    assert result.exit_code==0,result.output
    assert calls[-1]['output']==str(working/'chosen')
