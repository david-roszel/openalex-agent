"""Publication safeguards; synthetic values, no private credential access or APIs."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts.repository_audit import audit,local_only

@pytest.mark.parametrize('name',['credentials/grader.env','private-dist/grading.zip','developer.env','versions/revision7/data/cases.json','keys.private.json','questions-heldout.json'])
def test_private_submission_and_reserved_inputs_are_local(name):
    assert local_only(name)
    assert local_only("docs/submission/offline-mechanism-validation.json")


def test_blank_template_required_without_exposing_opaque_value(tmp_path):
    secret='SYNTHETIC_OPAQUE_TEMPLATE_VALUE'
    (tmp_path/'.env.example').write_text('LLM_API_KEY='+secret+'\n')
    result=audit(tmp_path)
    assert any(c['type']=='nonempty credential in configuration template' for c in result['concerns'])
    assert secret not in json.dumps(result)


def test_frozen_reference_matches_original_source_pins():
    root=Path(__file__).resolve().parents[1]/'versions/revision7'
    pins=json.loads((root/'SOURCE_MANIFEST.json').read_text())['source_pins']
    assert len(pins)==118
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in pins.items())
