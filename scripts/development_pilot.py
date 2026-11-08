"""Prepare/check an unscored pilot offline, or run only with exact explicit approval."""
import argparse
from pathlib import Path
from unittest.mock import patch
import httpx

from openalex_agent.config import Settings
from openalex_agent.development_pilot import execute,approval_pins,validate_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','offline','run'))
    p.add_argument('--spec',required=True)
    p.add_argument('--approval')
    p.add_argument('--output')
    a=p.parse_args();settings=Settings.from_env()
    if a.mode=='check':
        import json
        print(json.dumps(dict(pins=approval_pins(a.spec,settings),hosted_requests=0,authorization='PENDING'),indent=2));return
    if a.mode=='offline':
        def forbidden(*args,**kwargs):raise AssertionError('HTTP forbidden in development preflight')
        with patch.object(httpx.Client,'send',forbidden),patch.object(httpx.AsyncClient,'send',forbidden):
            rows=execute(a.spec,None,settings,offline=True,offline_output=a.output)
        print('OFFLINE ENGINEERING ONLY:',len(rows),'cells; hosted calls=0; OpenAlex calls=',sum(r['api_requests'] for r in rows))
    else:
        if not a.approval:p.error('run requires explicit --approval')
        rows=execute(a.spec,a.approval,settings)
        print('Unscored hosted development cells:',len(rows))


if __name__=='__main__':main()
