"""Synthetic engineering fixtures only; no model calls or scientific judgments."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import monitor


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    db = sqlite3.connect(root / 'queue.sqlite')
    db.execute('CREATE TABLE jobs(id TEXT,stage TEXT,payload TEXT,state TEXT,started REAL,ended REAL,error TEXT)')
    rows = [
        ('a','B-main','standard',3,'COMPLETED',1100,1200),
        ('b','B-main','llm',3,'FAILED',1200,1300),
        ('c','C-long','clm_graph',10,'INTERRUPTED',1300,1400),
        ('d','C-long','clm',10,'RUNNING',1400,None),
        ('pilot','A-pilot','llm_graph',3,'COMPLETED',1000,1050),
    ]
    for rid,stage,method,budget,state,started,ended in rows:
        payload={'task_id':'SYNTHETIC-'+rid,'method':method,'action_budget':budget,'repetition':0}
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(rid,stage,json.dumps(payload),state,started,ended,None))
    db.commit(); db.close()
    write_json(root / 'status.json', {'status':'running','stage':'B-main'})
    write_json(root / 'authorization.json', {'start_epoch':1000,'maximum_runtime_seconds':86400})
    write_json(root / 'spending-ledger.json', {
        'cumulative_retained_exposure_usd':'17.13072','ceiling_usd':'300',
        'confirmed_upper_bound_gross_usd':'1.00','prior_retained_exposure_usd':'14',
        'retained_usd':'3.13072','stop_reason':None,
        'attempts':[{'id':1,'status':'UNKNOWN','diagnostic':{'http_status':None}}]})


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        fixture(self.root)
        self.process = patch('monitor.process_status', return_value={'pid':123,'alive':True})
        self.process.start()

    def tearDown(self):
        self.process.stop()
        self.directory.cleanup()

    def test_terminal_failures_count_pilot_excluded_and_remaining_includes_running(self):
        s = monitor.build_snapshot(self.root, now=2000)
        self.assertEqual(s['overall']['terminal'],3)
        self.assertEqual(s['overall']['succeeded'],1)
        self.assertEqual(s['overall']['failed'],2)
        self.assertEqual(s['overall']['running'],1)
        self.assertEqual(s['overall']['remaining'],897)
        self.assertEqual(s['overall']['not_started'],896)
        self.assertAlmostEqual(s['overall']['percent'],100*3/900)
        self.assertEqual(s['latest_success']['id'],'a')
        self.assertEqual(s['latest_terminal']['id'],'c')

    def test_all_methods_and_budget_breakdowns(self):
        s = monitor.build_snapshot(self.root, now=2000)
        self.assertEqual(set(s['architectures']),set(monitor.METHODS))
        self.assertEqual(s['architectures']['llm_graph']['terminal'],0)
        self.assertEqual(s['architectures']['clm_graph']['failed'],1)
        self.assertEqual(s['action_budgets']['3']['terminal'],2)
        self.assertEqual(s['action_budgets']['10']['terminal'],1)

    def test_exposure_comes_from_conservative_ledger_not_successful_requests(self):
        s = monitor.build_snapshot(self.root, now=2000)
        self.assertEqual(s['requests']['http200'],0)
        self.assertEqual(s['spending']['cumulative_exposure_usd'],'17.13072')
        self.assertEqual(s['spending']['unknown_requests'],1)
        self.assertTrue(any('unresolved' in w for w in s['warnings']))

    def test_eta_wall_throughput_including_failures_and_pauses(self):
        s = monitor.build_snapshot(self.root, now=2000)
        self.assertEqual(s['runtime']['campaign_elapsed_seconds'],1000)
        self.assertEqual(s['runtime']['main_elapsed_seconds'],900)
        self.assertEqual(s['runtime']['terminal_per_hour'],12)
        self.assertEqual(s['runtime']['estimated_main_remaining_seconds'],269100)
        self.assertIn('Not a completion guarantee',s['runtime']['eta_assumptions'])

    def test_supervisor_paused_stopped_completed(self):
        write_json(self.root/'status.json',{'status':'stopped_by_safeguard','stop_reason':'campaign_unaccountable_usage'})
        self.assertEqual(monitor.build_snapshot(self.root,now=2000)['supervisor'],'paused')
        with patch('monitor.process_status',return_value={'pid':123,'alive':False}):
            self.assertEqual(monitor.build_snapshot(self.root,now=2000)['supervisor'],'stopped')
            write_json(self.root/'status.json',{'status':'complete_eligible_campaign'})
            write_json(self.root/'FINALIZED.json',{'execution_status':'complete_eligible_campaign'})
            self.assertEqual(monitor.build_snapshot(self.root,now=2000)['supervisor'],'completed')

    def test_source_files_unchanged_even_with_uncheckpointed_wal(self):
        db = sqlite3.connect(self.root/'queue.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA wal_autocheckpoint=0')
        db.execute("UPDATE jobs SET error='uncheckpointed engineering fixture' WHERE id='a'")
        db.commit()
        paths=[self.root/'queue.sqlite',self.root/'queue.sqlite-wal',self.root/'queue.sqlite-shm']
        def signatures():
            return [(p.name,p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
        before=signatures()
        rows=monitor.read_queue(self.root/'queue.sqlite')
        self.assertEqual(next(r for r in rows if r['id']=='a')['error'],'uncheckpointed engineering fixture')
        self.assertEqual(before,signatures())
        db.close()

    def test_no_active_sqlite_connection(self):
        original=sqlite3.connect
        seen=[]
        def connect(path,*a,**kw):
            self.assertNotEqual(Path(path).resolve(),(self.root/'queue.sqlite').resolve())
            seen.append(path)
            return original(path,*a,**kw)
        with patch('monitor.sqlite3.connect',side_effect=connect):
            monitor.read_queue(self.root/'queue.sqlite')
        self.assertTrue(seen)

    def test_html_is_local_self_refreshing_and_escaped(self):
        s=monitor.build_snapshot(self.root,now=2000)
        s['stage']='<script>unsafe</script>'
        page=monitor.html_view(s,30)
        self.assertIn('http-equiv="refresh" content="30"',page)
        self.assertIn('&lt;script&gt;unsafe&lt;/script&gt;',page)
        self.assertNotIn('<script>unsafe</script>',page)
        self.assertIn('STALE snapshot',page)
        self.assertNotIn('<script src=',page)
        self.assertIn('0.33%',page)

    def test_empty_initial_queue_renders_without_fabricated_results(self):
        db=sqlite3.connect(self.root/'queue.sqlite');db.execute('DELETE FROM jobs');db.commit();db.close()
        s=monitor.build_snapshot(self.root,now=2000)
        self.assertEqual(s['overall']['terminal'],0)
        self.assertIsNone(s['runtime']['estimated_main_remaining_seconds'])
        self.assertIn('none yet',monitor.html_view(s,30))

    def test_completed_runtime_stays_fixed_and_missing_cost_stays_unknown(self):
        write_json(self.root/'status.json',{'status':'complete_eligible_campaign','finished_epoch':1500})
        write_json(self.root/'FINALIZED.json',{'execution_status':'complete_eligible_campaign'})
        s=monitor.build_snapshot(self.root,now=5000)
        self.assertEqual(s['runtime']['campaign_elapsed_seconds'],500)
        self.assertEqual(monitor.money(None),'unavailable')

    def test_missing_ledger_does_not_invent_spending(self):
        (self.root/'spending-ledger.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'exposure cannot be inferred'):
            monitor.build_snapshot(self.root,now=2000)


if __name__=='__main__':
    unittest.main()
