"""Offline development worksheets; no automatic ground truth or API requests."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from .corpus import LocalCorpus, atomic_json, freeze

ROOT = Path(__file__).resolve().parent.parent


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(corpus_path, output, *, questions_path=None, factual_questions_path=None, candidates_path=None, reserved_path=None):
    """Prepare blank review worksheets; bundled defaults are synthetic examples.

    Supply your own public tasks/candidates and a local reserved-question list
    for real protocol preparation. This function never verifies human truth.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a fresh development-review directory")
    corpus = LocalCorpus(corpus_path)
    questions = json.loads(Path(questions_path or ROOT / "benchmarks/example_questions.json").read_text())
    factual = json.loads(Path(factual_questions_path or ROOT / "benchmarks/example_factual_questions.json").read_text())
    candidates = json.loads(Path(candidates_path or ROOT / "benchmarks/example_review_candidates.json").read_text())
    if not questions or len(questions) != len(factual) or len(questions) != len(candidates):
        raise ValueError("Supply one factual probe and review candidate per public question")
    if reserved_path is not None:
        excluded = json.loads(Path(reserved_path).read_text())
        if {q["question"].strip().casefold() for q in questions} & {q["question"].strip().casefold() for q in excluded}:
            raise ValueError("Public review questions overlap locally reserved questions")
    output.mkdir(parents=True)
    snapshot = freeze(corpus.papers, output / "corpus", acquisition=[{
        "source": "user-selected corpus copied without retrieval; inspect its acquisition provenance",
        "parent_sha256": corpus.fingerprint, "parent_directory": str(corpus_path)}])
    atomic_json(output / "questions.json", questions)
    atomic_json(output / "factual-questions.json", factual)
    sources = output / "sources"
    sources.mkdir()
    for paper in corpus.papers:
        atomic_json(sources / (paper.id.rsplit("/",1)[-1]+".json"), paper.to_dict())
    rows = []
    for q in questions:
        for p in corpus.papers:
            rows.append(dict(query_id=q["id"], question=q["question"], paper_id=p.id, title=p.title,
                available_source_type="abstract" if p.abstract else "metadata_only", source_material=p.abstract or "",
                source_location="sources/"+p.id.rsplit("/",1)[-1]+".json#abstract",
                source_sha256=hashlib.sha256((p.abstract or "").encode()).hexdigest(),
                grade="", supporting_passage="", explanation="", judgment_provenance="", assessor="",
                independent_human_review="", uncertainty="unreviewed", disagreement="unreviewed", adjudication_note=""))
    random.Random(43).shuffle(rows)  # no method/rank/relevance hints
    write_csv(output / "relevance-worksheet.csv", rows)
    targets = candidates
    adequacy = []
    worksheet = []
    for q, probe, candidate in zip(questions, factual, targets):
        target = candidate["work_id"].rsplit("/", 1)[-1]
        p = corpus.by_id.get("https://openalex.org/"+target)
        if p is None:
            raise ValueError("The selected development source snapshot lacks a required review candidate")
        preliminary = candidate["preliminary_adequacy"]
        if preliminary not in ("sufficient", "partially_sufficient", "insufficient"):
            raise ValueError("Invalid preliminary candidate adequacy")
        explanation = "Preliminary candidate supplied by the protocol author; independent human verification is pending."
        candidate = dict(query_id=q["id"], question_id=probe["question_id"], factual_question=probe["question"],
            candidate_work_id=p.id, candidate_title=p.title, candidate_passage=p.abstract or "",
            preliminary_adequacy=preliminary, preliminary_explanation=explanation,
            preliminary_provenance="unverified protocol input, NOT independent ground truth",
            reviewed_adequacy="", verified_answer="", supporting_work_id="", supporting_passage="",
            source_type="", evidence_location="", judgment_provenance="", assessor="", human_verification_status="pending",
            independent_human_review="", uncertainty="unreviewed", disagreement="unreviewed", adjudication_note="")
        worksheet.append(candidate)
        adequacy.append(dict(query_id=q["id"], question_id=probe["question_id"],
            candidate_work_id=p.id, available_source_type="abstract", preliminary_adequacy=preliminary,
            explanation=explanation, human_verification="pending", reviewed_adequacy="UNKNOWN",
            usable_for_factual_evaluation=False, verified_ground_truth_facts=[]))
    write_csv(output / "evidence-adequacy-worksheet.csv", worksheet)
    atomic_json(output / "evidence-adequacy.json", {"source_inspection_only":True,"questions":adequacy})
    manifest = {"schema_version":1,"development_only":True,"corpus_manifest":snapshot,
        "candidate_pool_policy":f"all {len(corpus.papers)} snapshot records for each question; relevance labels blank; order shuffled independently of retrieval methods",
        "questions":questions,"factual_questions":factual,"relevance_rows":len(rows),
        "independent_judgments":"UNKNOWN","evidence_verification":"UNKNOWN",
        "initial_immutable_sha256":{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*") if p.is_file() and not p.name.endswith("worksheet.csv")}}
    atomic_json(output / "review-manifest.json", manifest)
    return manifest


def validate_review(directory):
    """Structural readiness checks do not certify a reviewer's independence."""
    directory=Path(directory)
    manifest=json.loads((directory/"review-manifest.json").read_text())
    for name, digest in manifest["initial_immutable_sha256"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Immutable review input changed; create a new bundle")
    corpus=LocalCorpus(directory/"corpus")
    with (directory/"relevance-worksheet.csv").open(newline="",encoding="utf-8") as stream: relevance=list(csv.DictReader(stream))
    with (directory/"evidence-adequacy-worksheet.csv").open(newline="",encoding="utf-8") as stream: facts=list(csv.DictReader(stream))
    expected={(q["id"],p.id) for q in manifest["questions"] for p in corpus.papers}
    actual=[(r["query_id"],r["paper_id"]) for r in relevance]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError("Incomplete or duplicate relevance worksheet identity")
    factual_expected={q["question_id"] for q in manifest["factual_questions"]}
    if {r["question_id"] for r in facts}!=factual_expected or len(facts)!=len(factual_expected):raise ValueError("Factual worksheet identity mismatch")
    errors=[];pending=[]
    question_text={q["id"]:q["question"] for q in manifest["questions"]}
    factual_text={q["question_id"]:q for q in manifest["factual_questions"]}
    def verified(r):
        return (r.get("assessor","").strip() and r.get("judgment_provenance","").strip()
            and r.get("independent_human_review")=="true" and r.get("uncertainty")=="false"
            and r.get("disagreement")=="false")
    for r in relevance:
        if r["question"]!=question_text[r["query_id"]]:raise ValueError("Annotation question differs from immutable public question")
        if r["grade"].strip() and r["grade"] not in ("0","1","2"):errors.append("Development relevance grade must be 0, 1 or 2")
        if not r["grade"].strip() or not verified(r):pending.append(r["query_id"]+"/"+r["paper_id"]);continue
        if r["grade"] not in ("0","1","2") or not r["explanation"].strip():errors.append("Invalid grade or missing relevance explanation")
        if r["supporting_passage"] and r["supporting_passage"] not in (corpus.by_id[r["paper_id"]].abstract or ""):
            errors.append("Relevance passage not present in the provided abstract")
    for r in facts:
        task=factual_text[r["question_id"]]
        if r["query_id"]!=task["query_id"] or r["factual_question"]!=task["question"]:raise ValueError("Factual annotation question changed")
        if not verified(r) or r["human_verification_status"]!="verified":pending.append(r["question_id"]);continue
        if r["reviewed_adequacy"] not in ("sufficient","partially_sufficient","insufficient"):
            errors.append("Invalid reviewed adequacy")
        if r["reviewed_adequacy"]!="insufficient":
            paper=corpus.by_id.get(r["supporting_work_id"])
            if not r["verified_answer"].strip() or not paper or not r["supporting_passage"] or r["supporting_passage"] not in (paper.abstract or ""):
                errors.append("Answerable factual item requires a verified answer and an actual supporting passage")
            expected_location=("sources/"+paper.id.rsplit("/",1)[-1]+".json#abstract") if paper else None
            if r["source_type"]!="abstract" or r["evidence_location"]!=expected_location:errors.append("Missing/unsupported source type or location; curate any new full text in a new bundle")
        elif not r["adjudication_note"].strip():errors.append("Insufficiency requires an explanation of the evidence gap")
    return dict(independent_judgments="FAIL" if errors else "UNKNOWN" if pending else "PASS",
        evidence_adequacy="FAIL" if errors else "UNKNOWN" if pending else "PASS",
        pending_count=len(pending),errors=errors,
        independence_note="Reviewer attestations are required; software cannot establish actual human independence",
        recommendation="NOT READY FOR SMALL HOSTED PILOT" if errors or pending else "REQUIRES HUMAN PROTOCOL SIGN-OFF")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus")
    parser.add_argument("--output")
    parser.add_argument("--check")
    parser.add_argument("--questions")
    parser.add_argument("--factual-questions")
    parser.add_argument("--candidates")
    parser.add_argument("--reserved")
    args=parser.parse_args()
    if args.check:result=validate_review(args.check)
    elif args.corpus and args.output:result={"prepared":args.output,"rows":prepare(args.corpus,args.output,questions_path=args.questions,factual_questions_path=args.factual_questions,candidates_path=args.candidates,reserved_path=args.reserved)["relevance_rows"]}
    else:parser.error("Use --corpus/--output to prepare, or --check to validate")
    print(json.dumps(result))


if __name__=="__main__":main()
