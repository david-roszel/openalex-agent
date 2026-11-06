"""Public-release boundaries and generic protocol inputs; no empirical findings."""

import json
from pathlib import Path

import pytest

from openalex_agent.development_review import prepare, validate_review
from openalex_agent.models import Paper

ROOT = Path(__file__).resolve().parents[1]


def test_only_synthetic_public_example_questions_are_distributed():
    assert {p.name for p in (ROOT/'benchmarks').iterdir()} == {
        'README.md','example_questions.json','example_factual_questions.json','example_review_candidates.json'}
    questions=json.loads((ROOT/'benchmarks/example_questions.json').read_text())
    assert len(questions)==3 and all(q['split']=='diagnostic' for q in questions)
    records=[Paper.from_dict(json.loads(line)) for line in (ROOT/'tests/fixtures/synthetic-corpus/papers.jsonl').read_text().splitlines()]
    assert len(records)==39 and all(p.title.startswith('SYNTHETIC') for p in records)


def test_generic_review_inputs_preserve_blank_human_judgments_and_reserved_guard(tmp_path):
    questions=ROOT/'benchmarks/example_questions.json'
    factual=ROOT/'benchmarks/example_factual_questions.json'
    candidates=ROOT/'benchmarks/example_review_candidates.json'
    corpus=ROOT/'tests/fixtures/synthetic-corpus'
    reserved=tmp_path/'reserved.json'
    reserved.write_text(questions.read_text())
    with pytest.raises(ValueError,match='overlap'):
        prepare(corpus,tmp_path/'blocked',questions_path=questions,factual_questions_path=factual,candidates_path=candidates,reserved_path=reserved)
    assert not (tmp_path/'blocked').exists()
    reserved.write_text('[]')
    output=tmp_path/'review'
    prepare(corpus,output,questions_path=questions,factual_questions_path=factual,candidates_path=candidates,reserved_path=reserved)
    report=validate_review(output)
    assert report['pending_count']==120 and report['independent_judgments']=='UNKNOWN'
    assert report['evidence_adequacy']=='UNKNOWN'


def test_generic_freeze_cli_uses_only_explicit_source_mapping(tmp_path,monkeypatch):
    from openalex_agent import freeze_pilot
    source=tmp_path/'selected.txt';source.write_text('SYNTHETIC PUBLIC FIXTURE')
    mapping=tmp_path/'sources.json';mapping.write_text(json.dumps({'evidence.txt':str(source)}))
    output=tmp_path/'snapshot';receipt=tmp_path/'receipt.json'
    monkeypatch.setattr('sys.argv',['freeze','freeze','--receipt',str(receipt),'--output',str(output),'--sources',str(mapping)])
    freeze_pilot.main()
    assert freeze_pilot.verify(json.loads(receipt.read_text()))['files']==1
    assert (output/'evidence.txt').read_bytes()==source.read_bytes()
