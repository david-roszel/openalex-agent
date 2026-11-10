"""Offline publication controls; synthetic envelopes, never real credentials."""

import json
from pathlib import Path
import subprocess

import pytest

from scripts import repository_audit as scanner

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = [
    '.venv',
    'application_default_credentials.json',
    'gcloud/application_default_credentials.json',
    'benchmarks/questions-heldout.json', 'benchmarks/questions-heldout.csv',
    'benchmarks/human-annotations.csv', 'benchmarks/human-labels.jsonl',
    'keys.private.json', 'approval.json', 'nested/approval-template.json',
    'credentials/grader.env', '.streamlit/secrets.toml', 'service_account_local.json',
    'client_secret_local.json', 'id_ed25519', 'local.crt',
    'provider-requests.jsonl', 'hosted-responses.jsonl', 'spending-ledger.json',
    'request-receipts.jsonl', 'logs/provider.json', 'traces/research.json',
    'notebooks/private/work.ipynb', 'notebooks/run.private.ipynb',
    'notebooks/.ipynb_checkpoints/run.ipynb', 'grading.tgz',
]


@pytest.mark.parametrize('name', SENSITIVE)
def test_sensitive_names_are_ignored_and_rejected_from_exact_index(name):
    assert scanner.local_only(name)
    if not (ROOT / '.git').exists():
        pytest.skip('Git ignore integration requires a Git checkout')
    result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '--quiet', name],
                            capture_output=True, timeout=10)
    assert result.returncode == 0, name


@pytest.mark.parametrize('name', [
    '.env.example', 'benchmarks/example_questions.json',
    'benchmarks/live_submission_v1/questions.json',
    'tests/fixtures/development-corpus/papers.jsonl',
    'tests/fixtures/meta-context-failed-edits.json',
    'openalex_agent/tokenizer_data/o200k_base.tiktoken',
    'openalex_agent/_vendor/meta_clm/NOTICE', 'versions/revision7/SOURCE_MANIFEST.json',
    'docs/POOLED_RELEVANCE_PROTOCOL.md', 'docs/SECURITY_REVIEW.md',
])
def test_public_sources_and_fixtures_remain_candidates(name):
    assert not scanner.local_only(name)
    if not (ROOT / '.git').exists():
        pytest.skip('Git ignore integration requires a Git checkout')
    result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '--quiet', name],
                            capture_output=True, timeout=10)
    assert result.returncode == 1, name


@pytest.mark.parametrize('document', [
    {'type': 'service_account'}, {'type': 'authorized_user'}, {'type': 'external_account'},
    {'private_key': 'SYNTHETIC_NEVER_AUTHENTICATES'},
    {'refresh_token': 'SYNTHETIC_NEVER_AUTHENTICATES'},
    {'client_secret': 'SYNTHETIC_NEVER_AUTHENTICATES'},
    {'installed': {'client_secret': 'SYNTHETIC_NEVER_AUTHENTICATES'}},
    {'web': {'client_secret': 'SYNTHETIC_NEVER_AUTHENTICATES'}},
])
def test_renamed_credential_envelopes_refused_without_echo(tmp_path, monkeypatch, document):
    data = json.dumps(document).encode()
    monkeypatch.setattr(scanner, 'staged_candidates', lambda _: [('innocent.json', False, data)])
    report = scanner.audit(tmp_path, staged=True)
    assert {'path': 'innocent.json', 'type': 'provider credential JSON envelope'} in report['concerns']
    assert 'SYNTHETIC_NEVER_AUTHENTICATES' not in json.dumps(report)


def test_forced_staging_sensitive_names_remains_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, 'staged_candidates', lambda _: [(n, False, b'{}') for n in SENSITIVE])
    report = scanner.audit(tmp_path, staged=True)
    flagged = {c['path'] for c in report['concerns'] if c['type'] == 'local-only artifact included'}
    assert flagged == set(SENSITIVE)


@pytest.mark.parametrize('document', [[], {'type': 'research'}, {'files': {'tokens.py': 'checksum'}},
                                      {'installed': []}, {'web': 'documentation'}])
def test_scientific_and_provenance_json_not_mistaken_for_auth(tmp_path, monkeypatch, document):
    monkeypatch.setattr(scanner, 'staged_candidates', lambda _: [('research.json', False, json.dumps(document).encode())])
    assert not any(c['type'] == 'provider credential JSON envelope' for c in scanner.audit(tmp_path, staged=True)['concerns'])
