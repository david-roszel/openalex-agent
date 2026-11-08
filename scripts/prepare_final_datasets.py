"""Offline dataset intake; no API calls, packaged script execution or final runs."""

import argparse
from pathlib import Path
import json

from openalex_agent.corpus import atomic_json
from openalex_agent.dataset_integration import integrate_package, audit_corpora, prepare_annotations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    report=integrate_package(args.package,args.output)
    root=Path(__file__).resolve().parents[1]
    paths=[root/"data/frozen/three-question-pilot-v1/evidence/graph-ablation-pilot-2026-11-02/corpus",
           root/"data/corpus/live-smoke",root/"data/corpus/milestone3-real-cache"]
    audit,papers=audit_corpora([p for p in paths if p.is_dir()],(root/"data/ui-runs").glob("*/metadata.sqlite"))
    atomic_json(args.output/"real-corpus-audit.json",audit)
    prepare_annotations(args.package,papers,args.output/"annotations")
    print(json.dumps({k:report[k] for k in ("schema_validation","disciplines","question_splits","independently_verified_questions","acquired_real_papers","synthetic_papers","synthetic_edges","synthetic_probes","empirical_ready","heldout_text_exported")}))


if __name__=="__main__":main()
