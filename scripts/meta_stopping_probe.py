"""Prepare/check offline; run only after separate, pinned paid-inference approval."""
import argparse
import json

from openalex_agent.stopping_probe import prepare, verify, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--output', required=True)
    p = commands.add_parser('check'); p.add_argument('--spec', required=True)
    p = commands.add_parser('run'); p.add_argument('--spec', required=True); p.add_argument('--approval', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': result = prepare(args.output)
    elif args.mode == 'check': result = verify(args.spec)
    else: result = execute(args.spec, args.approval)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
