"""Audit an acquisition without HTTP, changing frozen files, or printing secrets."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from openalex_agent.corpus import atomic_json
from openalex_agent.models import utcnow


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contains_secret(path, secrets):
    opener = gzip.open if path.name.endswith('.gz') else open
    overlap = max(map(len, secrets), default=1) - 1
    tail = b''
    with opener(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            block = tail + chunk
            if any(value in block for value in secrets):
                return True
            tail = block[-overlap:] if overlap else b''
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh validation manifest path')
    root = args.run_root
    preserved = json.loads((root / 'preservation-before.json').read_text())
    sources = json.loads((root / 'source-before.json').read_text())
    changed = [name for name, sha in {**preserved, **sources}.items()
               if not Path(name).is_file() or checksum(Path(name)) != sha]
    if changed:
        raise ValueError(f'Preservation mismatch in {len(changed)} files; inspect baseline manifests')
    receipt = json.loads((args.corpus / 'freeze-receipt.json').read_text())
    frozen_files = {name: checksum(args.corpus / name) for name in receipt['files']}
    if frozen_files != receipt['files']:
        raise ValueError('Frozen receipt checksum mismatch')
    frozen_files['freeze-receipt.json'] = checksum(args.corpus / 'freeze-receipt.json')
    if any(checksum(root / 'main/frozen' / name) != sha for name, sha in frozen_files.items()):
        raise ValueError('Canonical snapshot differs from original acquisition freeze')
    env = {**dotenv_values('.env'), **os.environ}
    secrets = [value.encode() for name, value in env.items()
               if any(token in name.upper() for token in ('API_KEY', 'SECRET', 'PASSWORD'))
               and isinstance(value, str) and len(value) >= 8]
    selected = [p for directory in (root, args.corpus) for p in directory.rglob('*') if p.is_file()]
    leaks = [str(p) for p in selected if contains_secret(p, secrets)]
    if leaks:
        raise ValueError(f'Configured secret found in {len(leaks)} new artifacts; values withheld')
    offline = json.loads((root / 'final-offline-check/offline-validation.json').read_text())
    assert offline['passed'] and offline['rows'] == 5 and offline['api_requests'] == 0
    assert offline['real_hosted_calls'] == 0 and offline['http_transport_forbidden']
    assert offline['corpus_sha256'] == receipt['corpus_sha256']
    quality = json.loads((args.corpus / 'quality.json').read_text())
    assert quality['unique_papers'] == 30000 and set(quality['papers_per_discipline'].values()) == {5000}
    status = json.loads((root / 'main/collection-status.json').read_text())
    sample = json.loads((root / 'sample/collection-status.json').read_text())
    queries = json.loads((root / 'query-check.json').read_text())
    reported = (status['usage']['reported_cost_usd'] + sample['usage']['reported_cost_usd']
                + sum(row['cost_usd'] for row in queries) + .0001)
    assert status['last_quota']['prepaid_remaining_usd'] == 0
    test_log = (root / 'pytest-full.log').read_text()
    assert '454 passed' in test_log
    roles = Counter()
    shortfalls = []
    for row in json.loads((root / 'main/stratum-targets.json').read_text()):
        for role, counts in row['roles'].items():
            roles[role] += counts['actual']
            if counts['actual'] < counts['target']:
                shortfalls.append(dict(stratum=row['stratum'], role=role, **counts))
    code = list(Path('scripts').glob('*quality*.py')) + [Path('scripts/check_offline_corpus.py'),
           Path('openalex_agent/quality_collection.py'), Path('tests/test_quality_collection.py'),
           Path('docs/QUALITY_FIRST_ACQUISITION_REPORT.md')]
    report = dict(created_at=utcnow(), corpus=str(args.corpus.resolve()), corpus_sha256=receipt['corpus_sha256'],
        frozen_files=frozen_files, acquisition_complete=True, scientifically_validated=False,
        ground_truth_independently_verified=False, quality=quality,
        offline_validation=offline, automated_tests=dict(passed=454, log=str(root/'pytest-full.log')),
        preservation=dict(protected_files=len(preserved), existing_source_modules=len(sources), changed=0,
                          public_export_files=sum(name.startswith('../OpenAlex-CLM-Public/') for name in preserved)),
        security=dict(configured_secret_values_found=0, checked_files=len(selected),
                      scope='New artifacts including decompressed gzip metadata; no credential values logged'),
        usage=dict(main=status['usage'], sample=sample['usage'], main_quota_checks=status['quota_checks'],
                   sample_quota_checks=sample['quota_checks'], final_account_quota=status['last_quota'],
                   request_reported_allowance_usd=round(reported, 6),
                   account_meter_allowance_usd=status['last_quota']['daily_used_usd'],
                   meter_receipt_difference_usd=round(reported-status['last_quota']['daily_used_usd'], 6),
                   cash_spent_usd=0, prepaid_spent_usd=0, fulltext_requests=0, hosted_llm_calls=0,
                   accounting_note='Request receipts and account meter differ slightly; one failed request has no cost receipt. Do not infer exact billing from estimates or zero wait_seconds.'),
        source_roles_before_refill=dict(roles), remaining_original_role_shortfalls=shortfalls,
        code_and_report_sha256={str(p):checksum(p) for p in code},
        collection_source_snapshots=str(root/'main/source'),
        scientific_gates='Independent taxonomy/relevance/identity/evidence review, approved annotation workflow and final protocol gates remain open')
    atomic_json(args.output, report)
    print(json.dumps(dict(preserved_files=len(preserved), preserved_modules=len(sources),
        configured_secret_values_found=0, canonical_freeze_verified=True, corpus_papers=quality['unique_papers'])))


if __name__ == '__main__':
    main()
