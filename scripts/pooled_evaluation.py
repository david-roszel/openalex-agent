"""Explicit two-stage protocol; preparation is offline, collect requires approval."""
import argparse
from openalex_agent.pooled_review import prepare_pool,import_review,resolve_reviews,freeze_outputs,score_pool,collect_unscored


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--corpus',required=True);prep.add_argument('--questions',required=True)
    prep.add_argument('--frozen-outputs');prep.add_argument('--heldout-approval')
    imp=sub.add_parser('import');imp.add_argument('--pool',required=True);imp.add_argument('--worksheet',required=True)
    imp.add_argument('--attestation',required=True);imp.add_argument('--reviewer',required=True);imp.add_argument('--role',choices=('A','B'),required=True)
    res=sub.add_parser('resolve');res.add_argument('--pool',required=True);res.add_argument('--reviews',nargs='+',required=True)
    res.add_argument('--adjudication');res.add_argument('--attestation');res.add_argument('--adjudicator')
    freeze=sub.add_parser('freeze');freeze.add_argument('--experiment',required=True);freeze.add_argument('--approval',required=True)
    score=sub.add_parser('score');score.add_argument('--frozen-outputs',required=True);score.add_argument('--pool',required=True)
    score.add_argument('--annotations',required=True);score.add_argument('--approval',required=True)
    collect=sub.add_parser('collect');collect.add_argument('--spec',required=True);collect.add_argument('--question-review',required=True);collect.add_argument('--approval',required=True)
    for parser in (prep,imp,res,freeze,score):parser.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='prepare':result=prepare_pool(a.corpus,a.questions,a.output,frozen_outputs=a.frozen_outputs,heldout_approval=a.heldout_approval);print('Unreviewed candidates:',result['candidates'])
    elif a.command=='import':print('Imported rows:',len(import_review(a.pool,a.worksheet,a.attestation,a.reviewer,a.role,a.output)))
    elif a.command=='resolve':result=resolve_reviews(a.pool,a.reviews,a.output,adjudication=a.adjudication,attestation=a.attestation,adjudicator=a.adjudicator);print('Judgment coverage:',result['coverage'])
    elif a.command=='freeze':print('Frozen cells:',freeze_outputs(a.experiment,a.output,a.approval)['cells'])
    elif a.command=='score':print('Scored preserved cells:',len(score_pool(a.frozen_outputs,a.pool,a.annotations,a.approval,a.output)))
    elif a.command=='collect':
        from openalex_agent.config import Settings
        print('Collected cells:',len(collect_unscored(a.spec,a.question_review,a.approval,Settings.from_env())))


if __name__=='__main__':main()
