"""Prepare separate canary/comparison proposals; run only with explicit pinned approval."""
import argparse
import json
from pathlib import Path

from openalex_agent.config import Settings
from openalex_agent.meta_validation import prepare,execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=['canary','comparison'],default='canary')
    p.add_argument('--canary-status',type=Path)
    p=commands.add_parser('run');p.add_argument('--spec',required=True);p.add_argument('--approval',required=True)
    args=parser.parse_args();settings=Settings.from_env()
    if args.command=='prepare':
        spec=prepare(args.output,settings,mode=args.mode,canary_status=args.canary_status)
        print(json.dumps({k:spec[k] for k in ('mode','methods','trials','max_provider_calls','spending_ceiling_usd','checkpoint_usd')},indent=2))
    else:print(json.dumps(execute(args.spec,args.approval,settings),indent=2))


if __name__=='__main__':main()
