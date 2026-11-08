"""Offline intake for the supplied evaluation package; never execute its collector.

Held-out text stays in the original archive. Synthetic keys stay in an
evaluator-only directory. This adapter does not grant scientific approval.
"""

from collections import Counter
from contextlib import closing
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import zipfile
import xml.etree.ElementTree as ET

from .corpus import LocalCorpus, atomic_json, freeze
from .experiment_spec import BenchmarkSpec
from .models import Paper, canonical_id

QUESTION_FILE = "openalex_multidisciplinary_questions.csv"
PLAN_FILE = "openalex_corpus_collection_plan.csv"
PAPER_FILE = "openalex_synthetic_papers.csv"
EDGE_FILE = "openalex_synthetic_edges.csv"
PROBE_FILE = "openalex_synthetic_probes.csv"
VISIBLE_FILE = "openalex_synthetic_probes_model_visible.csv"
WORKBOOK = "OpenAlex_CLM_Multidisciplinary_Evaluation_Datasets.xlsx"
SPLITS = {"DEV": "development", "VALIDATION": "validation", "HELD-OUT": "heldout"}
FIXTURE_ID = re.compile(r"SYN-[A-Z]+-\d+-\d+")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique(rows, key):
    values = [r[key] for r in rows]
    if any(not v.strip() for v in values) or len(set(values)) != len(values):
        raise ValueError("Missing or duplicate " + key)


def _write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell_equal(key, left, right):
    if left == right:
        return True
    numeric = {"Synthetic score", "Synthetic year", "Theme index", "Local index",
               "Target unique OpenAlex works", "Actually acquired", "Verified usable works",
               "Keyword seed target", "Citation-neighbor target", "Field/background target", "Hard-negative target"}
    if key not in numeric:
        return False
    try:
        a,b = Decimal(left),Decimal(right)
        return a.is_finite() and b.is_finite() and a == b
    except (InvalidOperation, TypeError):
        return False


def _workbook_rows(content):
    """Standard-library XLSX reader; formulas are not executed or trusted."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(content)) as book:
        if sum(i.file_size for i in book.infolist()) > 100_000_000:
            raise ValueError("Oversized workbook")
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(t.text or "" for t in s.findall(".//m:t", ns))
                       for s in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        links = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))}
        sheets = {}
        for sheet in ET.fromstring(book.read("xl/workbook.xml")).findall("m:sheets/m:sheet", ns):
            target = links[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            for row in ET.fromstring(book.read(path)).findall(".//m:sheetData/m:row", ns):
                cells = {}
                for cell in row.findall("m:c", ns):
                    letters = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter) - 64
                    value = cell.find("m:v", ns)
                    text = value.text or "" if value is not None else "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                    cells[column-1] = strings[int(text)] if cell.attrib.get("t") == "s" else text
                rows.append([cells.get(i, "") for i in range(max(cells, default=-1)+1)])
            sheets[sheet.attrib["name"]] = rows
        return sheets


def load_package(path):
    """Validate CSV identities/relationships and workbook parity without printing text."""
    path = Path(path)
    members = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) != 1 or item.file_size > 50_000_000:
                raise ValueError("Unsafe or oversized package member")
            if item.filename in members:
                raise ValueError("Duplicate archive member")
            members[item.filename] = archive.read(item)
    tables = {}
    for name in (QUESTION_FILE, PLAN_FILE, PAPER_FILE, EDGE_FILE, PROBE_FILE, VISIBLE_FILE):
        reader = csv.DictReader(io.StringIO(members[name].decode("utf-8-sig")))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Invalid CSV columns")
        rows = list(reader)
        if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError("Invalid CSV row width")
        tables[name] = rows
    questions, plan, papers, edges, probes, visible = (tables[n] for n in (QUESTION_FILE, PLAN_FILE, PAPER_FILE, EDGE_FILE, PROBE_FILE, VISIBLE_FILE))
    for rows, key in ((questions, "Question ID"), (plan, "Stratum ID"), (papers, "Fixture work ID"), (probes, "Probe ID"), (visible, "Probe ID")):
        _unique(rows, key)
    strata = {r["Stratum ID"]: r for r in plan}
    question_texts = set()
    for row in questions:
        if row["Preassigned split"] not in SPLITS or not 1 <= len(row["Candidate question"].strip()) <= 1000:
            raise ValueError("Unknown question split or invalid question length")
        normalized = " ".join(row["Candidate question"].casefold().split())
        if normalized in question_texts:
            raise ValueError("Duplicate question text across package splits")
        question_texts.add(normalized)
        if row["Corpus stratum"] not in strata or row["Discipline"] != strata[row["Corpus stratum"]]["Discipline"]:
            raise ValueError("Question/stratum discipline mismatch")
        if row["Status"] != "UNVERIFIED CANDIDATE" or row["Human reviewer"].strip() or row["Review date"].strip():
            raise ValueError("This intake version accepts the unreviewed package, not changed human assertions")
    for row in plan:
        sizes = [int(row[k]) for k in ("Keyword seed target", "Citation-neighbor target", "Field/background target", "Hard-negative target")]
        if any(n < 0 for n in sizes) or sum(sizes) != int(row["Target unique OpenAlex works"]):
            raise ValueError("Collection strata allocations do not sum to target")
        if date.fromisoformat(row["Min date"]) > date.fromisoformat(row["Max date"]):
            raise ValueError("Invalid collection date interval")
        if int(row["Actually acquired"]) != 0 or int(row["Verified usable works"]) != 0:
            raise ValueError("Nonzero acquisition assertions need separate record validation")
    paper_ids = {r["Fixture work ID"] for r in papers}
    declared_edges = set()
    for row in papers:
        if not FIXTURE_ID.fullmatch(row["Fixture work ID"]) or row["Evidence origin"] != "SYNTHETIC / ENGINEERING ONLY":
            raise ValueError("Unlabeled or invalid synthetic paper")
        int(row["Synthetic year"])
        if not math.isfinite(float(row["Synthetic score"])):
            raise ValueError("Nonfinite synthetic score")
        refs = FIXTURE_ID.findall(row["Synthetic reference IDs"])
        if set(refs)-paper_ids:
            raise ValueError("Synthetic reference target is missing")
        declared_edges.update((row["Fixture work ID"], target) for target in refs)
    edge_pairs = []
    for row in edges:
        pair = (row["Citing synthetic work"], row["Cited synthetic work"])
        if row["Relation"] != "CITES" or row["Edge type"] != "SYNTHETIC" or set(pair)-paper_ids:
            raise ValueError("Invalid synthetic edge")
        edge_pairs.append(pair)
    if len(set(edge_pairs)) != len(edge_pairs) or set(edge_pairs) != declared_edges:
        raise ValueError("Synthetic edge table and paper references disagree")
    by_probe = {r["Probe ID"]: r for r in probes}
    visible_columns = set(visible[0])
    expected_visible = {"Probe ID", "Domain ID", "Discipline", "Theme", "Probe type", "Planned pressure", "Prompt / factual opportunity", "Pressure status", "Run status"}
    if visible_columns != expected_visible:
        raise ValueError("Unknown model-visible probe columns")
    if visible_columns & {"Correct answer or abstention", "Supporting synthetic evidence", "Ground-truth source work", "Ground-truth status"}:
        raise ValueError("Evaluator fields leaked into model-visible probes")
    if set(by_probe) != {r["Probe ID"] for r in visible}:
        raise ValueError("Probe identities disagree")
    for row in visible:
        full = by_probe[row["Probe ID"]]
        if any(full.get(k) != v for k, v in row.items()) or full["Ground-truth status"] != "SYNTHETIC GROUND TRUTH":
            raise ValueError("Model-visible probe projection differs")
        if set(FIXTURE_ID.findall(full["Ground-truth source work"]))-paper_ids:
            raise ValueError("Probe refers to an unknown synthetic source")
    sheets = _workbook_rows(members[WORKBOOK])
    for filename, sheet in ((QUESTION_FILE,"Question Bank"),(PLAN_FILE,"Corpus Plan"),(PAPER_FILE,"Synthetic Papers"),(EDGE_FILE,"Synthetic Edges"),(PROBE_FILE,"Synthetic Probes")):
        rows = sheets[sheet]
        header = rows[0]
        records = [dict(zip(header, r + [""]*(len(header)-len(r)))) for r in rows[1:]]
        if len(records) != len(tables[filename]) or any(any(not _cell_equal(k,record.get(k),v) for k,v in original.items()) for record,original in zip(records,tables[filename])):
            raise ValueError("Workbook/CSV mismatch in " + sheet)
    if "Review Queue" in sheets:
        header,*rows=sheets["Review Queue"]
        queue=[dict(zip(header,r+[""]*(len(header)-len(r)))) for r in rows]
        actual={r["Question ID"]:r["Preassigned split"] for r in questions}
        if len(queue)!=len(actual) or {r["Question ID"]:r["Split"] for r in queue}!=actual or any(int(r["Verified judgments"])!=0 for r in queue):
            raise ValueError("Workbook review queue differs from unreviewed question bank")
    if "Experiment Grid" in sheets:
        header,*rows=sheets["Experiment Grid"]
        grid=[dict(zip(header,r+[""]*(len(header)-len(r)))) for r in rows]
        _unique(grid,"Run ID")
        allowed={"OpenAlex baseline","Conventional LLM","Graph LLM","CLM LLM","Graph+CLM LLM"}
        if any(r["Architecture"] not in allowed or r["Planned pressure"] not in ("Low","Medium","High") or int(r["Seed"])<0
               or int(r["Active context cap"])>=int(r["Total input cap"]) or r["Execution status"]!="PLANNED ONLY / NO RESULTS"
               or int(r["Actual input tokens"])!=0 for r in grid):
            raise ValueError("Invalid planned experiment grid or unsupported result assertion")
    return dict(path=path, sha256=digest(path.read_bytes()), tables=tables, sheets=sheets,
                member_sha256={k:digest(v) for k,v in members.items()})


def integrate_package(package, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh dataset integration directory")
    data = load_package(package)
    output.mkdir(parents=True)
    tables = data["tables"]
    questions = tables[QUESTION_FILE]
    for split in ("development", "validation"):
        rows = [{"id":r["Question ID"],"question":r["Candidate question"],"split":split,
                 "discipline":r["Discipline"],"stratum":r["Corpus stratum"],"review_status":"unverified"}
                for r in questions if SPLITS[r["Preassigned split"]] == split]
        atomic_json(output/split/"questions.json", rows)
    heldout = [{"id":r["Question ID"],"discipline":r["Discipline"],"stratum":r["Corpus stratum"]}
               for r in questions if r["Preassigned split"] == "HELD-OUT"]
    atomic_json(output/"sealed/heldout-index.json", {"text_location":"Original package only; explicit approved final execution required","questions":heldout})
    atomic_json(output/"evaluation-only/synthetic-probes.json", tables[PROBE_FILE])
    atomic_json(output/"diagnostics/model-visible-probes.json", tables[VISIBLE_FILE])
    # Canonical-shaped aliases are a mock API interface, never real OpenAlex IDs.
    papers = tables[PAPER_FILE]
    aliases = {r["Fixture work ID"]:canonical_id("W"+str(990000000000+i)) for i,r in enumerate(papers,1)}
    normalized = []
    for row in papers:
        raw = {"id":aliases[row["Fixture work ID"]],"title":"SYNTHETIC ["+row["Fixture work ID"]+"] "+row["Synthetic title"],
               "publication_year":int(row["Synthetic year"]),"referenced_works":[aliases[k] for k in FIXTURE_ID.findall(row["Synthetic reference IDs"])]}
        paper = Paper.from_openalex(raw, provenance="SYNTHETIC PACKAGE MOCK; NOT AN OPENALEX OBSERVATION")
        paper.abstract = row["Synthetic abstract / evidence"]
        normalized.append(paper)
    freeze(normalized,output/"diagnostics/corpus",acquisition=[{"source":"SYNTHETIC PACKAGE MOCK ONLY; NO LIVE OPENALEX ACCESS",
        "package_sha256":data["sha256"],"identity_warning":"Canonical-shaped aliases are not verified OpenAlex IDs; never merge into real corpus"}])
    atomic_json(output/"diagnostics/synthetic-id-aliases.json", {"not_real_openalex_ids":True,"mapping":aliases})
    atomic_json(output/"diagnostics/questions.json",[{"id":r["Probe ID"],"question":r["Prompt / factual opportunity"],"split":"diagnostic","discipline":r["Discipline"]} for r in tables[VISIBLE_FILE]])
    atomic_json(output/"diagnostics/pilot-questions.json",[{"id":r["Probe ID"],"question":r["Prompt / factual opportunity"],"split":"diagnostic","discipline":r["Discipline"]} for r in tables[VISIBLE_FILE][:3]])
    atomic_json(output/"diagnostics/factual-questions.json",[{"query_id":r["Probe ID"],"question_id":r["Probe ID"],"question":r["Prompt / factual opportunity"]} for r in tables[VISIBLE_FILE]])
    plan = tables[PLAN_FILE]
    atomic_json(output/"acquisition-proposal.json", {"approved":False,"package_sha256":data["sha256"],"target_unique_real_papers":sum(int(r["Target unique OpenAlex works"]) for r in plan),
        "actually_acquired_by_package":0,"strata":plan,"execution":"NOT AUTHORIZED; use trusted client after budget/protocol approval; packaged collector not executed"})
    report = {"package_sha256":data["sha256"],"member_sha256":data["member_sha256"],"schema_validation":"PASS",
        "disciplines":dict(Counter(r["Discipline"] for r in questions)),"question_splits":dict(Counter(SPLITS[r["Preassigned split"]] for r in questions)),
        "independently_verified_questions":0,"acquired_real_papers":0,"synthetic_papers":len(papers),"synthetic_edges":len(tables[EDGE_FILE]),"synthetic_probes":len(tables[PROBE_FILE]),
        "workbook_csv_parity":"PASS","workbook_sheets":list(data["sheets"]),"empirical_ready":False,"heldout_text_exported":False,
        "output_sha256":{str(p.relative_to(output)):digest(p.read_bytes()) for p in output.rglob("*") if p.is_file()}}
    atomic_json(output/"integration.json",report)
    return report


def development_spec(corpus, questions, output, **kwargs):
    """Resolve only explicit non-held-out inputs into the existing runner spec."""
    rows = json.loads(Path(questions).read_text())
    if not rows or any(r.get("split") not in ("development", "validation", "diagnostic") for r in rows):
        raise ValueError("Held-out or undeclared splits are not permitted for a development spec")
    if len({r["split"] for r in rows}) != 1:
        raise ValueError("Do not combine development, validation or diagnostic splits")
    if len({r["id"] for r in rows}) != len(rows) or any(not isinstance(r.get("question"),str) or not 1<=len(r["question"])<=1000 for r in rows):
        raise ValueError("Invalid benchmark question schema")
    LocalCorpus(corpus)
    return BenchmarkSpec(str(corpus),str(questions),str(output),**kwargs)


def corpus_statistics(papers):
    """Statistics only; records and graph infrastructure are never modified."""
    ids = {p.id for p in papers}
    if len(ids)!=len(papers) or any(canonical_id(p.id)!=p.id for p in papers):
        raise ValueError("Duplicate or noncanonical audited paper IDs")
    internal = {(p.id,r) for p in papers for r in p.references if r in ids}
    refs = {r for p in papers for r in p.references}
    years = [p.year for p in papers if p.year is not None]
    dates = [p.publication_date for p in papers if p.publication_date]
    dois = Counter(p.doi for p in papers if p.doi)
    return dict(unique_papers=len(ids),abstracts=sum(bool(p.abstract and p.abstract.strip()) for p in papers),
        missing_titles=sum(not p.title for p in papers),missing_doi=sum(not p.doi for p in papers),missing_year=sum(p.year is None for p in papers),
        publication_dates_present=len(dates),date_min=min(dates,default=None),date_max=max(dates,default=None),year_min=min(years,default=None),year_max=max(years,default=None),
        internal_reference_edges=len(internal),all_reference_edges=sum(len(p.references) for p in papers),unresolved_reference_targets=len(refs-ids),
        isolated_from_internal_citations=len(ids-{x for edge in internal for x in edge}),duplicate_doi_groups=sum(n>1 for n in dois.values()),
        field_mapping_status="Not assigned to package disciplines; requires taxonomy mapping and independent audit")


def audit_corpora(paths, ui_databases=()):
    seen = {}
    snapshots = []
    for path in paths:
        corpus = LocalCorpus(path)
        if any("SYNTHETIC" in (p.provenance+" "+(p.title or "")).upper() for p in corpus.papers):
            raise ValueError("Synthetic data cannot enter the real corpus audit")
        snapshots.append(dict(path=str(path),sha256=corpus.fingerprint,**corpus_statistics(corpus.papers)))
        for p in corpus.papers:
            seen.setdefault(p.id,[]).append(p)
    snapshot_union = len(seen)
    db_records = []
    raw_domains = {}
    for path in ui_databases:
        path = Path(path)
        config=json.loads((path.parent/"research.json").read_text()).get("configuration",{})
        if config.get("mode")!="live":
            continue
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)) as connection:
            papers=[Paper.from_dict(json.loads(row[0])) for row in connection.execute("select payload from papers")]
            papers=[p for p in papers if p.provenance=="OpenAlex API" and "SYNTHETIC" not in (p.title or "").upper()]
            for row in connection.execute("select payload from cache"):
                payload=json.loads(row[0])
                for raw in payload.get("results",[payload]):
                    if isinstance(raw.get("primary_topic"),dict):
                        raw_domains[raw.get("id")]=raw["primary_topic"].get("domain",{}).get("display_name")
        for p in papers:
            seen.setdefault(p.id,[]).append(p)
        db_records.append(dict(path=str(path),**corpus_statistics(papers)))
    # Select the latest recorded version for statistics only; no corpus is built.
    selected=[max(observations,key=lambda p:p.retrieved_at) for observations in seen.values()]
    report=dict(snapshots=snapshots,snapshot_union_unique=snapshot_union,live_ui_databases=db_records,all_audited_real_records=corpus_statistics(selected),
        raw_primary_domain_counts=dict(Counter(d for i,d in raw_domains.items() if i in seen and d)),raw_primary_domain_records=sum(i in seen and bool(d) for i,d in raw_domains.items()),
        versioned_ids=sum(len(v)>1 for v in seen.values()),independently_verified_relevance_labels=0,independently_verified_factual_items=0,
        note="Stored acquisition provenance, not a new online verification; caches are not an approved final corpus")
    return report, selected


def prepare_annotations(package, papers, output):
    """Blank human worksheets for non-held-out questions; no suggested answers."""
    output=Path(output)
    if output.exists():
        raise ValueError("Annotation directory already exists")
    data=load_package(package)
    questions=[r for r in data["tables"][QUESTION_FILE] if r["Preassigned split"]!="HELD-OUT"]
    output.mkdir(parents=True)
    _write_csv(output/"question-review.csv",[dict(query_id=r["Question ID"],split=SPLITS[r["Preassigned split"]],discipline=r["Discipline"],question=r["Candidate question"],
        question_validity="",corpus_adequacy="",human_reviewer="",second_reviewer="",independent_human_review="",adjudication="",status="PENDING") for r in questions])
    _write_csv(output/"paper-relevance.csv",[dict(query_id=r["Question ID"],paper_id=p.id,grade="",assessor="",supporting_passage="",explanation="",independent_human_review="",uncertainty="",disagreement="",adjudication_note="") for r in questions for p in papers])
    _write_csv(output/"source-material.csv",[dict(paper_id=p.id,title=p.title or "",abstract=p.abstract or "",source_url=p.source_url or "",acquired_at=p.retrieved_at,
        passage_sha256=digest((p.abstract or "").encode()),proposed_discipline="",verified_discipline="",reviewer="") for p in papers])
    _write_csv(output/"factual-evidence.csv",[dict(query_id=r["Question ID"],question_id="",factual_question="",verified_answer="",supporting_work_id="",exact_supporting_passage="",source_type="",evidence_location="",evidence_sufficiency="",citation_attribution_correct="",appropriate_abstention="",human_reviewer="",second_reviewer="",independent_human_review="",adjudication="",status="PENDING") for r in questions])
    atomic_json(output/"annotation-manifest.json",dict(package_sha256=data["sha256"],questions=len(questions),papers=len(papers),relevance_rows=len(questions)*len(papers),
        status="UNREVIEWED INTAKE; NOT A READY REVIEW BUNDLE",heldout_included=False,verified_answers_supplied=False,
        coverage="All locally audited papers, not a representative multidisciplinary pool. Refresh/version after acquisition.",
        files={p.name:digest(p.read_bytes()) for p in output.glob("*.csv")}))
