"""Prepare unapproved Gemini/ADC canary or comparison; execution requires pinned approval."""
import argparse
import json
from pathlib import Path

from openalex_agent.gemini import GeminiSettings
from openalex_agent.gemini_validation import GeminiValidationBackend
from openalex_agent.meta_validation import prepare,execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--sdk-python',default='~/gemini-test-env/bin/python')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--mode',choices=['canary','comparison'],default='canary')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--canary-status',type=Path)
    p=commands.add_parser('run');p.add_argument('--spec',required=True);p.add_argument('--approval',required=True)
    args=parser.parse_args();settings=GeminiSettings(project=args.project,sdk_python=args.sdk_python)
    backend=GeminiValidationBackend()
    if args.command=='prepare':
        value=prepare(args.output,settings,mode=args.mode,canary_status=args.canary_status,backend=backend)
        print(json.dumps({k:value[k] for k in ('protocol','mode','methods','max_provider_calls','spending_ceiling_usd','checkpoint_usd')},indent=2))
    else:print(json.dumps(execute(args.spec,args.approval,settings,backend=backend),indent=2))


if __name__=='__main__':main()
