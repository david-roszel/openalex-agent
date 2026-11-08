"""Offline intake correctness; invented miniature fixtures are not research data."""

import csv
import io
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as E

import pytest

from openalex_agent.dataset_integration import (load_package, integrate_package, development_spec,
    prepare_annotations, corpus_statistics, _cell_equal, QUESTION_FILE, PLAN_FILE, PAPER_FILE, EDGE_FILE, PROBE_FILE, VISIBLE_FILE, WORKBOOK)
from openalex_agent.corpus import LocalCorpus
from openalex_agent.models import Paper


def package_file(tmp_path, mutate=None):
    questions=[{'Question ID':str(i),'Candidate question':text,'Preassigned split':split,'Discipline':'TEST',
       'Corpus stratum':'T1','Status':'UNVERIFIED CANDIDATE','Human reviewer':'','Review date':''}
        for i,(split,text) in enumerate([('DEV','Synthetic development task'),('VALIDATION','Synthetic validation task'),('HELD-OUT','SEALED FIXTURE NEVER EXPORT')])]
    plan=[{'Stratum ID':'T1','Discipline':'TEST','Keyword seed target':'2','Citation-neighbor target':'0','Field/background target':'0',
       'Hard-negative target':'0','Target unique OpenAlex works':'2','Min date':'2016-01-01','Max date':'2026-10-31','Actually acquired':'0','Verified usable works':'0'}]
    papers=[{'Fixture work ID':'SYN-T-01-001','Discipline':'TEST','Evidence origin':'SYNTHETIC / ENGINEERING ONLY','Synthetic year':'2020','Synthetic score':'1.0',
             'Synthetic reference IDs':'SYN-T-01-002','Synthetic title':'fixture first','Synthetic abstract / evidence':'Public fixture evidence'},
            {'Fixture work ID':'SYN-T-01-002','Discipline':'TEST','Evidence origin':'SYNTHETIC / ENGINEERING ONLY','Synthetic year':'2021','Synthetic score':'2.0',
             'Synthetic reference IDs':'','Synthetic title':'fixture second','Synthetic abstract / evidence':'Second public fixture evidence'}]
    edges=[{'Citing synthetic work':'SYN-T-01-001','Cited synthetic work':'SYN-T-01-002','Relation':'CITES','Edge type':'SYNTHETIC'}]
    visible=[{'Probe ID':'P1','Domain ID':'T','Discipline':'TEST','Theme':'fixture','Probe type':'Fact recovery','Planned pressure':'Low',
             'Prompt / factual opportunity':'Synthetic public factual task','Pressure status':'PENDING','Run status':'CREATED / NOT RUN'}]
    probes=[{**visible[0],'Ground-truth source work':'SYN-T-01-001','Correct answer or abstention':'EVALUATOR KEY ONLY','Supporting synthetic evidence':'Public fixture evidence','Ground-truth status':'SYNTHETIC GROUND TRUTH'}]
    tables=dict(zip((QUESTION_FILE,PLAN_FILE,PAPER_FILE,EDGE_FILE,PROBE_FILE,VISIBLE_FILE),(questions,plan,papers,edges,probes,visible)))
    if mutate:mutate(tables)
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main';rn='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    wb=E.Element('workbook',xmlns=ns);sheets=E.SubElement(wb,'sheets');rels=E.Element('Relationships')
    xlsx=io.BytesIO()
    with zipfile.ZipFile(xlsx,'w') as book:
        for i,(filename,name) in enumerate([(QUESTION_FILE,'Question Bank'),(PLAN_FILE,'Corpus Plan'),(PAPER_FILE,'Synthetic Papers'),(EDGE_FILE,'Synthetic Edges'),(PROBE_FILE,'Synthetic Probes')],1):
            E.SubElement(sheets,'sheet',name=name,sheetId=str(i),attrib={'{'+rn+'}id':'rId'+str(i)})
            E.SubElement(rels,'Relationship',Id='rId'+str(i),Target='worksheets/sheet'+str(i)+'.xml')
            sheet=E.Element('worksheet',xmlns=ns);sd=E.SubElement(sheet,'sheetData');rows=tables[filename];header=list(rows[0])
            for ri,values in enumerate([header]+[[r[k] for k in header] for r in rows],1):
                row=E.SubElement(sd,'row',r=str(ri))
                for ci,value in enumerate(values):
                    cell=E.SubElement(row,'c',r=chr(65+ci)+str(ri),t='inlineStr');E.SubElement(E.SubElement(cell,'is'),'t').text=value
            book.writestr('xl/worksheets/sheet'+str(i)+'.xml',E.tostring(sheet))
        book.writestr('xl/workbook.xml',E.tostring(wb));book.writestr('xl/_rels/workbook.xml.rels',E.tostring(rels))
    path=tmp_path/'package.zip'
    with zipfile.ZipFile(path,'w') as archive:
        for name,rows in tables.items():
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);archive.writestr(name,stream.getvalue())
        archive.writestr(WORKBOOK,xlsx.getvalue())
    return path


def test_intake_seals_holdout_and_oracle_and_preserves_explicit_synthetic_identity(tmp_path):
    package=package_file(tmp_path);out=tmp_path/'out';report=integrate_package(package,out)
    assert report['question_splits']=={'development':1,'validation':1,'heldout':1} and not report['empirical_ready']
    assert not report['heldout_text_exported']
    assert all('SEALED FIXTURE NEVER EXPORT' not in p.read_text() for p in out.rglob('*.json'))
    for directory in ('development','validation','diagnostics'):
        assert all('EVALUATOR KEY ONLY' not in p.read_text() for p in (out/directory).rglob('*') if p.is_file())
    corpus=LocalCorpus(out/'diagnostics/corpus')
    assert len(corpus.papers)==2 and all('SYNTHETIC' in p.provenance for p in corpus.papers)
    assert json.loads((out/'diagnostics/synthetic-id-aliases.json').read_text())['not_real_openalex_ids']
    spec=development_spec(out/'diagnostics/corpus',out/'development/questions.json',tmp_path/'experiment')
    assert spec.suite=='controlled-five' and spec.backend_kwargs()['output']==str(tmp_path/'experiment')
    with pytest.raises(ValueError,match='fresh'):integrate_package(package,out)


@pytest.mark.parametrize('case',['duplicate_question','bad_split','unknown_stratum','bad_allocation','acquired_claim','edge_mismatch','oracle_leak'])
def test_schema_errors_fail_before_any_output(tmp_path,case):
    def mutate(t):
        if case=='duplicate_question':t[QUESTION_FILE][1]['Candidate question']=t[QUESTION_FILE][0]['Candidate question']
        if case=='bad_split':t[QUESTION_FILE][0]['Preassigned split']='TRAIN'
        if case=='unknown_stratum':t[QUESTION_FILE][0]['Corpus stratum']='missing'
        if case=='bad_allocation':t[PLAN_FILE][0]['Keyword seed target']='3'
        if case=='acquired_claim':t[PLAN_FILE][0]['Actually acquired']='30000'
        if case=='edge_mismatch':t[EDGE_FILE][0]['Cited synthetic work']='SYN-T-01-001'
        if case=='oracle_leak':t[VISIBLE_FILE][0]['Correct answer or abstention']='EVALUATOR KEY ONLY'
    package=package_file(tmp_path,mutate)
    with pytest.raises(ValueError):integrate_package(package,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_holdout_and_undeclared_inputs_cannot_resolve_development_spec(tmp_path):
    path=tmp_path/'questions.json'
    for split in ('heldout','proposed_heldout',None):
        path.write_text(json.dumps([dict(id='q',question='Unseen task',split=split)]))
        with pytest.raises(ValueError,match='Held-out'):development_spec('missing-corpus',path,tmp_path/'out')


def test_annotations_are_blank_and_never_include_heldout(tmp_path):
    package=package_file(tmp_path);paper=Paper.from_openalex({'id':'W1','title':'SYNTHETIC TEST'})
    out=tmp_path/'annotations';prepare_annotations(package,[paper],out)
    with (out/'paper-relevance.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(not r['grade'] and not r['assessor'] for r in rows)
    with (out/'factual-evidence.csv').open() as f:rows=list(csv.DictReader(f))
    assert all(not r['verified_answer'] and not r['exact_supporting_passage'] for r in rows)
    assert all('SEALED FIXTURE NEVER EXPORT' not in p.read_text() for p in out.iterdir())


def test_numeric_workbook_representation_is_not_textual_parity_bypass():
    assert _cell_equal('Synthetic score','1','1.0')
    assert not _cell_equal('Candidate question','1','1.0')
    assert not _cell_equal('Synthetic score','1','2')
    assert not _cell_equal('Synthetic score','NaN','nan')


def test_corpus_stats_never_count_unresolved_targets_as_acquired_papers():
    p=Paper.from_openalex({'id':'W1','referenced_works':['W2','W3']})
    report=corpus_statistics([p])
    assert report['unique_papers']==1 and report['unresolved_reference_targets']==2
    assert report['abstracts']==0 and report['internal_reference_edges']==0
    with pytest.raises(ValueError):corpus_statistics([p,p])


def test_nonfinite_package_scores_are_rejected(tmp_path):
    package=package_file(tmp_path,lambda t:t[PAPER_FILE][0].update({'Synthetic score':'NaN'}))
    with pytest.raises(ValueError,match='Nonfinite'):load_package(package)


def test_validation_and_development_cannot_be_combined_into_one_spec(tmp_path):
    path=tmp_path/'mixed.json';path.write_text(json.dumps([
        dict(id='d',question='Development fixture',split='development'),
        dict(id='v',question='Validation fixture',split='validation')]))
    with pytest.raises(ValueError,match='combine'):development_spec('missing',path,tmp_path/'output')
