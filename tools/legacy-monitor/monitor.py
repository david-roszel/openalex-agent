#!/usr/bin/env python3
"""Read-only OpenAlex benchmark monitor. Standard library; no provider/CLI calls.

The active SQLite DB is NEVER opened by SQLite: stable copies of its DB/WAL are
queried in a disposable directory, avoiding even writes to the active SHM file.
Only the monitor's own output directory receives persistent writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import html
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPOSITORY_ROOT / 'data/legacy-objective-benchmark'
METHODS = ('standard', 'llm', 'llm_graph', 'clm', 'clm_graph')
MAIN_STAGES = ('B-main', 'C-long')
TERMINAL = {'COMPLETED', 'FAILED', 'INTERRUPTED'}
FAILED = {'FAILED', 'INTERRUPTED'}
PLANNED = 900
LOCAL_ZONE = ZoneInfo('America/New_York')


def read_json(path: Path, default=None):
    try:
        with path.open() as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


def file_identity(path: Path):
    try:
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except FileNotFoundError:
        return None


def read_queue(path: Path):
    """Read a stable DB+WAL copy; never connect to active SQLite or read SHM."""
    last_error = None
    for _ in range(5):
        try:
            with tempfile.TemporaryDirectory(prefix='openalex-monitor-') as directory:
                copy = Path(directory) / 'queue.sqlite'
                source_files = [path, Path(str(path) + '-wal')]
                before = [file_identity(p) for p in source_files]
                if before[0] is None:
                    raise FileNotFoundError(path)
                for source, identity in zip(source_files, before):
                    if identity is not None:
                        shutil.copyfile(source, copy if source == path else Path(str(copy) + '-wal'))
                if before != [file_identity(p) for p in source_files]:
                    raise RuntimeError('Queue changed during snapshot; retrying')
                # SQLite may create local SHM/WAL files, only in this disposable copy.
                with sqlite3.connect(copy) as db:
                    db.execute('PRAGMA query_only=ON')
                    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                        raise RuntimeError('Snapshot integrity check failed')
                    db.row_factory = sqlite3.Row
                    return [dict(r) for r in db.execute('SELECT id,stage,payload,state,started,ended,error FROM jobs')]
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            last_error = exc
            time.sleep(.05)
    raise RuntimeError(f'Could not obtain a consistent read-only queue snapshot: {last_error}')


def process_status(benchmark: Path):
    try:
        pid = int((benchmark / 'supervisor-pid.txt').read_text().strip())
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text()
        state = stat.rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        matching = str(benchmark / 'tools/unattended.py') in command
        return {'pid': pid, 'alive': state not in ('Z', 'X') and matching,
                'identity_matches': matching, 'os_state': state}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': locals().get('pid'), 'alive': False, 'identity_matches': False}
    except (OSError, ValueError, IndexError) as exc:
        return {'pid': locals().get('pid'), 'alive': None, 'error': type(exc).__name__}


def summarize_group(rows, planned=None):
    states = Counter(r['state'] for r in rows)
    total = len(rows) if planned is None else planned
    terminal = sum(states[s] for s in TERMINAL)
    running = states['RUNNING']
    return {'planned': total, 'terminal': terminal, 'succeeded': states['COMPLETED'],
            'failed': sum(states[s] for s in FAILED), 'interrupted': states['INTERRUPTED'],
            'running': running, 'remaining': max(0, total - terminal),
            'not_started': max(0, total - terminal - running),
            'percent': 100 * terminal / total if total else 0,
            'unknown_states': {s: n for s, n in states.items() if s not in TERMINAL | {'PENDING', 'RUNNING'}}}


def job_label(row):
    if row is None:
        return None
    p = row['_payload']
    return {'id': row['id'], 'task': p.get('task_id', '?'), 'method': p.get('method', '?'),
            'action_budget': p.get('action_budget'), 'repetition': p.get('repetition'),
            'state': row['state'], 'ended': row.get('ended')}


def build_snapshot(benchmark: Path, *, now=None):
    now = time.time() if now is None else now
    warnings = []
    queue = read_queue(benchmark / 'queue.sqlite')
    rows = [r for r in queue if r['stage'] in MAIN_STAGES]
    for row in rows:
        row['_payload'] = json.loads(row['payload'])
    if len(rows) != PLANNED:
        warnings.append(f'Expected {PLANNED} main queue rows; found {len(rows)}. Denominator remains {PLANNED}.')
    totals = summarize_group(rows, PLANNED)
    if totals['unknown_states']:
        warnings.append(f"Unrecognized queue states: {totals['unknown_states']}")
    ledger = read_json(benchmark / 'spending-ledger.json')
    if not isinstance(ledger, dict):
        raise RuntimeError('Conservative spending ledger unavailable; exposure cannot be inferred')
    # Authoritative conservative exposure includes old retained usage and current holds.
    exposure = ledger.get('cumulative_retained_exposure_usd')
    ceiling = ledger.get('ceiling_usd')
    if exposure is None or ceiling is None:
        raise RuntimeError('Cumulative exposure or ceiling missing from spending ledger')
    state = read_json(benchmark / 'status.json', {})
    auth = read_json(benchmark / 'authorization.json', {})
    checkpoint = read_json(benchmark / 'FINALIZED.json')
    partial = read_json(benchmark / 'CHECKPOINT.json', {}) if checkpoint is None else {}
    proc = process_status(benchmark)
    stop = ledger.get('stop_reason') or state.get('stop_reason')
    complete = bool(checkpoint) and state.get('status') == 'complete_eligible_campaign'
    if complete:
        supervisor = 'completed'
    elif proc['alive'] is True:
        supervisor = 'paused' if stop or state.get('status') == 'stopped_by_safeguard' else 'running'
    elif proc['alive'] is False:
        supervisor = 'stopped'
    else:
        supervisor = 'unknown'
        warnings.append('Supervisor process identity could not be verified.')
    if totals['terminal'] == PLANNED and not complete:
        warnings.append('Main 900-cell queue is terminal; later stages/finalization may still be running.')
    starts = [r['started'] for r in rows if r.get('started') is not None]
    campaign_start = auth.get('start_epoch')
    # Freeze displayed runtime after completion; refresh time keeps advancing.
    runtime_now = now
    if complete:
        runtime_now = state.get('finished_epoch', now)
        if runtime_now == now and checkpoint.get('finalized_utc'):
            runtime_now = datetime.fromisoformat(checkpoint['finalized_utc']).timestamp()
    elapsed = max(0, runtime_now - campaign_start) if campaign_start is not None else None
    main_elapsed = max(0, now - min(starts)) if starts else None
    throughput = totals['terminal'] / main_elapsed if main_elapsed and totals['terminal'] else None
    eta = totals['remaining'] / throughput if throughput else None
    if totals['remaining'] == 0:
        eta = 0
    by_method = {m: summarize_group([r for r in rows if r['_payload'].get('method') == m], PLANNED // 5) for m in METHODS}
    budgets = sorted({r['_payload'].get('action_budget') for r in rows if isinstance(r['_payload'].get('action_budget'), int)})
    by_budget = {str(b): summarize_group([r for r in rows if r['_payload'].get('action_budget') == b]) for b in budgets}
    successes = [r for r in rows if r['state'] == 'COMPLETED' and r.get('ended') is not None]
    terminals = [r for r in rows if r['state'] in TERMINAL and r.get('ended') is not None]
    attempts = ledger.get('attempts', [])
    http = Counter(str((r.get('diagnostic') or {}).get('http_status')) for r in attempts)
    unknown = sum(r.get('status') == 'UNKNOWN' for r in attempts)
    reserved = sum(r.get('status') == 'RESERVED' for r in attempts)
    if unknown:
        warnings.append(f'{unknown} request(s) have unresolved usage; their conservative holds remain included.')
    receipt = None
    for attempt in reversed(attempts):
        if attempt.get('status') == 'RESERVED':
            continue
        path = benchmark / 'provider' / f"attempt-{attempt['id']:05d}.worker.json"
        value = read_json(path)
        if isinstance(value, dict):
            receipt = {'attempt': attempt['id'], 'http_status': value.get('http_status'),
                       'sdk_version': value.get('sdk_version'), 'path': str(path)}
        break
    budget10_started = any(r['_payload'].get('action_budget') == 10 and r.get('started') is not None for r in rows)
    if not budget10_started:
        warnings.append('10-action trajectories have not started; ETA extrapolates the shorter horizon and may be optimistic.')
    return {'updated_epoch': now, 'benchmark': str(benchmark), 'supervisor': supervisor,
            'process': proc, 'stage': state.get('stage'), 'stop_reason': stop,
            'overall': totals, 'architectures': by_method, 'action_budgets': by_budget,
            'spending': {'cumulative_exposure_usd': str(exposure), 'ceiling_usd': str(ceiling),
                         'percent': float(Decimal(str(exposure)) / Decimal(str(ceiling)) * 100),
                         'new_confirmed_conservative_gross_usd': ledger.get('confirmed_upper_bound_gross_usd'),
                         'prior_retained_exposure_usd': ledger.get('prior_retained_exposure_usd'),
                         'new_retained_exposure_usd': ledger.get('retained_usd'),
                         'reserved_requests': reserved, 'unknown_requests': unknown,
                         'ledger_updated_epoch': (benchmark / 'spending-ledger.json').stat().st_mtime},
            'requests': {'new_attempts': len(attempts), 'http200': http['200'], 'http429': http['429'],
                         'last_receipt': receipt},
            'runtime': {'campaign_elapsed_seconds': elapsed, 'main_elapsed_seconds': main_elapsed,
                        'terminal_per_hour': throughput * 3600 if throughput else None,
                        'estimated_main_remaining_seconds': eta,
                        'estimated_main_finish_epoch': now + eta if eta is not None and supervisor == 'running' else None,
                        'deadline_epoch': campaign_start + auth.get('maximum_runtime_seconds', 86400) if campaign_start else None,
                        'eta_assumptions': 'Observed main terminal throughput including pauses/failures; excludes later mechanism/public stages. Longer horizons and changing provider latency may take more time. Not a completion guarantee.'},
            'latest_success': job_label(max(successes, key=lambda r: r['ended'])) if successes else None,
            'latest_terminal': job_label(max(terminals, key=lambda r: r['ended'])) if terminals else None,
            'running_jobs': [job_label(r) for r in rows if r['state'] == 'RUNNING'],
            'checkpoint': {'finalized': bool(checkpoint), 'last_partial_status': partial.get('execution_status'),
                           'last_partial_utc': partial.get('finalized_utc')},
            'warnings': warnings}


def duration(seconds):
    if seconds is None:
        return 'not yet estimable'
    minutes = max(0, int(seconds)) // 60
    return f'{minutes // 60}h {minutes % 60:02d}m'


def money(value):
    return 'unavailable' if value is None else f'${Decimal(str(value)):.6f}'


def timestamp(epoch):
    return datetime.fromtimestamp(epoch, LOCAL_ZONE).strftime('%b %d %I:%M:%S %p %Z') if epoch is not None else 'unavailable'


def label(job):
    if job is None:
        return 'none yet'
    return f"{job['task']} | {job['method']} | {job['action_budget']} actions | repetition {job['repetition'] + 1 if job['repetition'] is not None else '?'}"


def progress_bar(percent, width=40):
    filled = min(width, max(0, int(percent * width / 100)))
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def terminal_view(s):
    o, cost, runtime = s['overall'], s['spending'], s['runtime']
    lines = ['OPENALEX SCIENTIFIC BENCHMARK — READ-ONLY MONITOR',
             f"Updated {timestamp(s['updated_epoch'])} | Supervisor: {s['supervisor'].upper()} | Stage: {s['stage']}",
             f"{progress_bar(o['percent'])} {o['percent']:.2f}%  ({o['terminal']}/{o['planned']} terminal)",
             f"Succeeded {o['succeeded']} | Failed {o['failed']} (interrupted {o['interrupted']}) | Running {o['running']}",
             f"Remaining {o['remaining']} including running | Not started {o['not_started']}", '',
             f"Cumulative conservative exposure: ${Decimal(cost['cumulative_exposure_usd']):.6f} / ${Decimal(cost['ceiling_usd']):.2f} ({cost['percent']:.2f}%)",
             f"New confirmed conservative estimate: {money(cost['new_confirmed_conservative_gross_usd'])} | Active reservations {cost['reserved_requests']} | Unknown usage {cost['unknown_requests']}",
             'Exposure includes prior retained usage and in-flight/unknown holds; it is not an invoice or credit balance.',
             f"Elapsed campaign {duration(runtime['campaign_elapsed_seconds'])} | Main throughput {runtime['terminal_per_hour']:.1f}/hour" if runtime['terminal_per_hour'] is not None else f"Elapsed campaign {duration(runtime['campaign_elapsed_seconds'])} | Throughput unavailable",
             f"Estimated remaining MAIN queue: {duration(runtime['estimated_main_remaining_seconds'])} (not guaranteed)",
             f"Campaign deadline: {timestamp(runtime['deadline_epoch'])}", '',
             'ARCHITECTURE       TERMINAL/PLAN   SUCCESS  FAILED  RUNNING  REMAINING']
    for name, g in s['architectures'].items():
        lines.append(f"{name:<18} {g['terminal']:>3}/{g['planned']:<6} {g['succeeded']:>7} {g['failed']:>7} {g['running']:>8} {g['remaining']:>10}")
    lines += ['', 'ACTION BUDGET      TERMINAL/PLAN   SUCCESS  FAILED  RUNNING  REMAINING']
    for name, g in s['action_budgets'].items():
        lines.append(f"{name + ' actions':<18} {g['terminal']:>3}/{g['planned']:<6} {g['succeeded']:>7} {g['failed']:>7} {g['running']:>8} {g['remaining']:>10}")
    lines += ['', 'Latest successful task: ' + label(s['latest_success']),
              'Completed at: ' + timestamp((s['latest_success'] or {}).get('ended')),
              'Running: ' + ('; '.join(label(r) for r in s['running_jobs']) or 'none'),
              f"Gemini receipts: {s['requests']['http200']} HTTP200 / {s['requests']['new_attempts']} attempts (this campaign)"]
    if s['stop_reason']:
        lines.append('Safeguard/status reason: ' + str(s['stop_reason']))
    lines += ['', 'ETA: ' + runtime['eta_assumptions']]
    lines += ['NOTE: ' + warning for warning in s['warnings']]
    lines += ['', 'Refresh every 30 seconds by default. Ctrl+C stops ONLY this monitor.']
    return '\n'.join(lines)


def html_view(s, interval):
    esc = lambda value: html.escape(str(value))
    o = s['overall']; cost = s['spending']; runtime = s['runtime']
    def table(groups):
        body = ''.join('<tr><th>' + esc(name) + '</th>' + ''.join('<td>' + esc(g[k]) + '</td>' for k in ('terminal','planned','succeeded','failed','running','remaining')) + '</tr>' for name, g in groups.items())
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + x + '</th>' for x in ('Method / budget','Terminal','Planned','Success','Failed','Running','Remaining')) + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    cards = ''.join(f'<article><span>{esc(title)}</span><strong>{esc(value)}</strong></article>' for title, value in [
        ('Success',o['succeeded']),('Failed / interrupted',o['failed']),('Running',o['running']),('Remaining (includes running)',o['remaining']),
        ('Cumulative exposure',f"${Decimal(cost['cumulative_exposure_usd']):.4f} / ${Decimal(cost['ceiling_usd']):.0f}"),
        ('Campaign elapsed',duration(runtime['campaign_elapsed_seconds'])),('Estimated main queue remaining',duration(runtime['estimated_main_remaining_seconds']))])
    notes = ''.join('<li>' + esc(w) + '</li>' for w in s['warnings'])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="{interval}"><title>OpenAlex benchmark progress</title>
<style>:root{{color-scheme:light dark;--bg:#F5F0E5;--panel:#fffdf7;--text:#27352d;--muted:#526457;--line:#c6cebf;--green:#244b3b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:32px 20px}}h1{{font-size:30px;margin:8px 0}}p{{line-height:1.6}}.muted,article span{{color:var(--muted)}}.bar{{height:22px;background:#c6cebf;border-radius:12px;overflow:hidden}}.bar div{{height:100%;background:#466a50}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}article{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;display:flex;flex-direction:column;gap:12px}}article strong{{font-size:25px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:var(--panel);font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child{{text-align:left}}.notice{{border-left:4px solid #a7794c;padding:10px 16px;background:var(--panel)}}#freshness{{font-weight:600}}@media(prefers-color-scheme:dark){{:root{{--bg:#17201b;--panel:#222e26;--text:#f5f0e5;--muted:#bdcbb8;--line:#4c5e4f}}}}</style></head><body><main>
<p class="muted">Read-only local monitor · no API calls · America/New_York</p><h1>OpenAlex scientific benchmark</h1><p><b>{esc(s['supervisor'].upper())}</b> · Stage {esc(s['stage'])} · Updated {esc(timestamp(s['updated_epoch']))}</p><p id="freshness"></p>
<h2>{o['percent']:.2f}% · {o['terminal']} / {o['planned']} terminal trajectories</h2><div class="bar" role="progressbar" aria-valuenow="{o['terminal']}" aria-valuemin="0" aria-valuemax="{o['planned']}"><div style="width:{min(100,o['percent']):.3f}%"></div></div>
{cards}<p class="muted">Terminal includes preserved failures. Remaining includes currently running trajectories. Pilot and supplemental stages are excluded from the900-cell denominator.</p>
<h2>Architectures</h2>{table(s['architectures'])}<h2>Action budgets</h2>{table(s['action_budgets'])}
<h2>Latest activity</h2><p>Most recent success: <b>{esc(label(s['latest_success']))}</b><br>{esc(timestamp((s['latest_success'] or {}).get('ended')))}</p><p>Running: {esc('; '.join(label(r) for r in s['running_jobs']) or 'none')}</p>
<p>HTTP200 responses: {s['requests']['http200']} / {s['requests']['new_attempts']} campaign attempts. Reservations: {cost['reserved_requests']}; unresolved usage: {cost['unknown_requests']}.</p>
<div class="notice"><p>Conservative spending exposure comes directly from the ledger, including prior retained amounts and active/unknown request holds. It is not an invoice or a promotional-credit balance.</p><p>{esc(runtime['eta_assumptions'])}</p><p>Deadline: {esc(timestamp(runtime['deadline_epoch']))}. {esc(s['stop_reason'] or '')}</p><ul>{notes}</ul></div>
<p class="muted">This page reloads every {interval} seconds. Keep the Python monitor running to refresh the underlying snapshot. Closing this page or stopping the monitor does not stop the experiment.</p></main>
<script>const updated={s['updated_epoch']};function freshness(){{const age=Math.max(0,Math.floor(Date.now()/1000-updated));const el=document.getElementById('freshness');el.textContent=age>{max(90,interval*3)}?'STALE snapshot — monitor may be stopped ('+age+' seconds old)':'Snapshot age: '+age+' seconds';}}freshness();setInterval(freshness,1000);</script></body></html>'''


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--output-dir', type=Path, default=REPOSITORY_ROOT / 'data/legacy-monitor-output')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true', help='Read, display and generate HTML once; then exit')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')
    benchmark, output = args.benchmark.resolve(), args.output_dir.resolve()
    if output == benchmark or benchmark in output.parents:
        parser.error('Monitor output must be outside the benchmark directory')
    output.mkdir(parents=True, exist_ok=True)
    last_good = None
    try:
        while True:
            try:
                snapshot = build_snapshot(benchmark)
                atomic_write(output / 'snapshot.json', json.dumps(snapshot, indent=2) + '\n')
                atomic_write(output / 'dashboard.html', html_view(snapshot, args.interval))
                last_good = snapshot
                view = terminal_view(snapshot)
            except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError) as exc:
                view = 'Monitor could not refresh: ' + str(exc)
                if last_good:
                    view += '\nDisplaying LAST GOOD snapshot; timestamps and HTML age show staleness.\n\n' + terminal_view(last_good)
                if args.once:
                    print(view, file=sys.stderr)
                    return 1
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(view, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nMonitor stopped. The benchmark was not signalled or modified.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
