"""Regenerate the public file inventory after explicit source edits; never stages or pushes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json
from repository_audit import source_candidates,audit
ROOT=Path(__file__).resolve().parents[1]

def main():
    checked=audit(ROOT)
    if checked['concerns']:raise SystemExit('Publication audit has concerns; manifest not rewritten. Run repository_audit.py for redacted details.')
    entries=[]
    for name,path in source_candidates(ROOT):
        if name=='PUBLIC_MANIFEST.json':continue
        if path.is_symlink():raise SystemExit('Symlink refused: '+name)
        raw=path.read_bytes();entries.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    result={'schema':2,'release':'Live Research, primary-five Scientific Evaluation and isolated frozen R7 reference',
      'prepared_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),'base_public_commit':'9b6d564f1e2f1c162ad487710947aa2c1530ce54',
      'published_commit':None,'git_history_preserved':True,'credentials_included':False,'private_benchmarks_included':False,
      'human_annotations_included':False,'raw_hosted_traces_included':False,'full_corpus_included':False,
      'small_authentic_and_synthetic_fixtures_included':True,'development_question_manifest_included':True,
      'audited_aggregate_status_included':True,'empirical_graph_clm_improvement_established':False,
      'checksum_scope':'All publication candidates except this manifest; generated/ignored runtime data excluded; no circular self-checksum',
      'included_files':sorted(entries,key=lambda x:x['path'])}
    (ROOT/'PUBLIC_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(entries),'bytes':sum(x['bytes'] for x in entries),'staged':False,'pushed':False}))

if __name__=='__main__':main()
