"""Command-line academic retrieval interface."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import sqlite3
from .config import Settings
from .models import Paper
from .openalex import OpenAlexClient, RetrievalError
from .storage import Database


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--planner", choices=["deterministic", "hosted"], default="deterministic")
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-inspected", type=int, default=200)
    parser.add_argument("--max-api-requests", type=int, default=50)
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=16000)


def run_budget(args):
    from .controller import Budget
    return Budget(max_actions=args.max_actions, max_rounds=args.max_rounds, max_inspected=args.max_inspected,
                  max_api_requests=args.max_api_requests, max_llm_calls=args.max_llm_calls,
                  page_size=args.page_size, output_k=args.top_k, patience=args.patience,
                  max_context_chars=args.context_chars)


def planner_for(args, settings):
    from .llm import DeterministicPlanner, HostedJSONPlanner
    return HostedJSONPlanner(settings) if args.planner == "hosted" else DeterministicPlanner()


def advanced_command(args, db: Database, client: OpenAlexClient, settings: Settings) -> bool:
    """Return whether an extended command was handled."""
    from .corpus import LocalCorpus, acquire, freeze
    if args.command == "corpus":
        if args.corpus_command == "build":
            manifest = acquire(client, args.queries, args.output, per_query=args.per_query, filters=args.filter)
        elif args.corpus_command == "freeze":
            manifest = freeze(db.papers(), args.output, acquisition=[{"source": "SQLite metadata cache", "database": settings.database}])
        else:
            manifest = LocalCorpus(args.directory).manifest
        print(json.dumps({"count": manifest["count"], "sha256": manifest["sha256"]}))
    elif args.command in ("agent", "baseline"):
        from .controller import Controller
        from .selection import Scoring
        retriever = LocalCorpus(args.corpus) if args.corpus else client
        if args.corpus:
            print(f"Retrieval: fixed corpus ({len(retriever.papers)} papers). "
                  "Live OpenAlex retrieval is unavailable; searches and expansions stay within this snapshot. "
                  "Rerun without --corpus for live access.", file=sys.stderr)
        else:
            print("Retrieval: live OpenAlex access enabled; cached responses may be reused.", file=sys.stderr)
        planner = planner_for(args, settings)
        try:
            result = Controller(args.question, retriever, db, method=args.method, planner=planner,
                budget=run_budget(args), scoring=Scoring(min_relevance=args.min_relevance, max_depth=args.max_depth,
                exploration=args.exploration, relevance=args.relevance_weight, cost=args.cost_weight,
                novelty=0 if args.no_novelty else args.novelty_weight,
                importance=0 if args.no_importance else args.importance_weight,
                coverage=0 if args.no_coverage else args.coverage_weight),
                output=args.output, resume=args.resume).run()
        finally:
            if hasattr(planner, "close"):
                planner.close()
        display([Paper.from_dict(item["paper"]) for item in result["ranked"]])
        for item in result["ranked"]:
            print(json.dumps({"id": item["paper"]["id"], **item["algorithmic"], "limitations": item["limitations"]}))
        print(json.dumps({k: result[k] for k in ["run_id", "planner", "model", "retrieval", "stop_reason", "actions", "api_usage", "llm_usage", "directory"]}))
        if result["stop_reason"] == "retrieval_error":
            raise SystemExit(1)
    elif args.command == "benchmark":
        print("Retrieval: fixed corpus. Live OpenAlex retrieval is unavailable; "
              "all benchmark methods use the same snapshot.", file=sys.stderr)
        from .experiments import run_experiments
        from .experiment_spec import BenchmarkSpec
        spec=BenchmarkSpec.from_cli(args)
        rows = run_experiments(spec.corpus, spec.questions, db, lambda: planner_for(args, settings),**spec.backend_kwargs())
        print(json.dumps({"runs": len(rows), "output": args.output, "has_judgments": bool(args.labels)}))
    elif args.command == "judgments-import":
        from .evaluation import import_judgments
        result = import_judgments(args.csv, args.output, scope=args.scope)
        print(json.dumps({"judgments": len(result["judgments"]), "scope": result["scope"]}))
    elif args.command == "claim-pool":
        from .research_assessment import build_claim_pool
        print(json.dumps({"claims_to_judge": build_claim_pool(args.experiment, args.output)}))
    elif args.command == "score-research":
        from .research_assessment import score_research
        print(json.dumps({"runs_scored": score_research(args.experiment, labels_path=args.labels,
            subtopics_path=args.subtopics, claim_judgments_path=args.claim_judgments)}))
    elif args.command == "evaluate":
        from .evaluation import load_judgments, metrics
        from .corpus import atomic_json
        result = json.loads((Path(args.run) / "result.json").read_text(encoding="utf-8"))
        judgments, scope = load_judgments(args.labels)
        if args.query_id not in judgments:
            raise ValueError("No judgments for this query ID")
        values = metrics([p["paper"]["id"] for p in result["ranked"]], judgments[args.query_id],
                         k=args.top_k, scope=scope, inspected=result["inspected_ids"], actions=result["actions"])
        atomic_json(Path(args.output), {"run_id": result["run_id"], "query_id": args.query_id, **values})
        print(json.dumps(values))
    elif args.command == "pool":
        from .experiments import build_pool
        print(json.dumps({"papers_to_judge": build_pool(args.experiment, args.output)}))
    elif args.command == "score-experiment":
        from .experiments import score_experiment
        print(json.dumps({"runs_scored": score_experiment(args.experiment, args.labels)}))
    elif args.command == "report":
        from .experiments import report
        print(json.dumps({"artifacts": report(args.experiment, final=args.final)}))
    elif args.command == "export":
        import shutil
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.events:
            events = db.events(args.events)
            path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        else:
            shutil.copyfile(Path(args.run) / ("ranked.csv" if args.format == "csv" else "result.json"), path)
        print(str(path))
    else:
        return False
    return True


def export_papers(papers: list[Paper], output: str) -> None:
    """Write normalized metadata atomically, retaining a previous export on failure."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps([p.to_dict() for p in papers], indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def graph_command(args, db: Database, client: OpenAlexClient) -> None:
    """Build, update, inspect or explicitly expand a scholarly graph."""
    from .graph import ScholarlyGraph
    if bool(args.expand) != bool(args.paper):
        raise ValueError("Use --expand references/citations together with --paper OPENALEX_ID")
    graph = ScholarlyGraph.load(args.load) if args.load else ScholarlyGraph()
    if not args.load or args.update:
        for paper in db.papers():
            old = graph.graph.nodes.get(paper.id, {}) if args.load else {}
            graph.add(paper, action="cache_update" if args.load else "cache_import",
                      parent=old.get("parent"), depth=old.get("depth", 0))
    expansion = None
    if args.expand:
        from .graph_expansion import expand_graph
        expansion = expand_graph(graph, client, args.paper, direction=args.expand, limit=args.limit)
        db.event(client.run_id, "graph_expansion", expansion)
    # Validate inspection options before writing a snapshot.
    view = graph.neighborhood([args.seed], hops=args.hops, max_nodes=args.max_nodes,
                              direction=args.direction, relations=args.relation) if args.seed else None
    discovery = graph.path(args.path) if args.path else None
    destination = args.output or (args.load if args.update or args.expand else None)
    if not args.load and not destination:
        destination = "data/graphs/cached.json"
    if destination:
        graph.save(destination)
    print(json.dumps({"run_id": client.run_id, **graph.detailed_stats(), "validated": True,
                      "output": destination, "usage": asdict(client.usage)}))
    if expansion is not None:
        print(json.dumps({"expansion": expansion}))
    if view is not None:
        print(json.dumps({"neighborhood": view}, ensure_ascii=False))
    if discovery is not None:
        print(json.dumps({"discovery_path": discovery}))


def display(papers: list[Paper], *, details: bool = False) -> None:
    if not papers:
        print("No papers found.")
    for index, paper in enumerate(papers, 1):
        authors = "; ".join(a.author.name or a.author.id if a.author else a.raw_name or "[author unknown]"
                            for a in paper.authorships)
        print(f"{index}. {paper.title or '[title unavailable]'} ({paper.year or 'year unavailable'})")
        print(f"   {authors or '[authors unavailable]'}")
        print(f"   {paper.id} | DOI: {paper.doi or 'unavailable'}")
        print(f"   {paper.venue or '[venue unavailable]'} | Abstract: {'available' if paper.abstract else 'unavailable'}")
        if details:
            print(f"   Source URL: {paper.source_url or 'unavailable'}")
            print(f"   Publication date: {paper.publication_date or 'unavailable'} | Type: {paper.work_type or 'unavailable'}")
            print(f"   Bibliographic fields: {json.dumps(paper.biblio, ensure_ascii=False)}")
            print(f"   Institutions: {'; '.join(dict.fromkeys(i.name or i.id for a in paper.authorships for i in a.institutions)) or 'unavailable'}")
            print(f"   Topics: {'; '.join(t.name or t.id for t in paper.topics) or 'unavailable'}")
            print(f"   Citation count: {paper.citation_count if paper.citation_count is not None else 'unavailable'}")
            print(f"   References: {', '.join(paper.references) or 'unavailable'}")
            print(f"   Related works: {', '.join(paper.related) or 'unavailable'}")
            print(f"   Retrieved at: {paper.retrieved_at} | Provenance: {paper.provenance}")
            print(f"   Abstract: {paper.abstract or '[abstract unavailable]'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-guided academic literature discovery")
    parser.add_argument("--db", help="SQLite metadata/cache path")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Search genuine OpenAlex metadata")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--filter", default="", help="OpenAlex filter expression, e.g. publication_year:2023-2026")
    search.add_argument("--refresh", action="store_true")
    search.add_argument("--output", help="Export normalized JSON")
    get = commands.add_parser("get", help="Inspect a work by OpenAlex ID")
    get.add_argument("paper_id")
    get.add_argument("--refresh", action="store_true")
    get.add_argument("--output", help="Export normalized metadata as a JSON list")
    commands.add_parser("cached", help="List persisted papers without network access")
    graph = commands.add_parser("graph", help="Build a graph from persisted papers, or inspect a saved graph")
    graph.add_argument("--load", help="Load a graph JSON instead of building from cached metadata")
    graph.add_argument("--output", help="Write a snapshot; new builds default to data/graphs/cached.json")
    graph.add_argument("--seed", help="OpenAlex ID for a bounded neighborhood")
    graph.add_argument("--hops", type=int, default=1)
    graph.add_argument("--max-nodes", type=int, default=50)
    graph.add_argument("--direction", choices=["in", "out", "both"], default="both")
    graph.add_argument("--relation", action="append", choices=["PAPER_CITES_PAPER", "PAPER_WRITTEN_BY_AUTHOR", "PAPER_HAS_TOPIC", "AUTHOR_AFFILIATED_WITH_INSTITUTION"])
    graph.add_argument("--path", help="Display the discovery path of a paper")
    graph.add_argument("--update", action="store_true", help="Merge the current SQLite metadata into a loaded graph")
    graph.add_argument("--expand", choices=["references", "citations"], help="Manually expand one scholarly relationship")
    graph.add_argument("--paper", help="Resolved OpenAlex paper ID used as the expansion anchor")
    graph.add_argument("--limit", type=int, default=20, help="Expansion result limit, 1–100")
    for name in ("agent", "baseline"):
        run = commands.add_parser(name, help="Run graph agent or comparison method")
        run.add_argument("question")
        run.add_argument("--method", choices=["standard", "reformulated", "iterative", "graph"], default="graph" if name == "agent" else "standard")
        run.add_argument("--corpus", help="Frozen corpus directory; omit for live OpenAlex")
        run.add_argument("--output", default="data/results")
        run.add_argument("--resume", help="Interrupted run directory; retain original configuration")
        run.add_argument("--min-relevance", type=float, default=0.03)
        run.add_argument("--max-depth", type=int, default=3)
        run.add_argument("--exploration", type=float, default=0.15)
        run.add_argument("--relevance-weight", type=float, default=0.45)
        run.add_argument("--novelty-weight", type=float, default=0.15)
        run.add_argument("--importance-weight", type=float, default=0.20)
        run.add_argument("--coverage-weight", type=float, default=0.20)
        run.add_argument("--cost-weight", type=float, default=0.10)
        run.add_argument("--no-novelty", action="store_true")
        run.add_argument("--no-importance", action="store_true")
        run.add_argument("--no-coverage", action="store_true")
        add_run_options(run)
    corpus = commands.add_parser("corpus", help="Acquire, freeze, or verify a fixed collection")
    corpora = corpus.add_subparsers(dest="corpus_command", required=True)
    build = corpora.add_parser("build")
    build.add_argument("queries", nargs="+")
    build.add_argument("--per-query", type=int, default=1000)
    build.add_argument("--filter", default="")
    build.add_argument("--output", required=True)
    cached = corpora.add_parser("freeze")
    cached.add_argument("--output", required=True)
    check = corpora.add_parser("check")
    check.add_argument("directory")
    benchmark = commands.add_parser("benchmark", help="Legacy four-method or controlled five-method fixed-corpus experiments")
    benchmark.add_argument("--corpus", required=True)
    benchmark.add_argument("--questions", default="benchmarks/example_questions.json")
    benchmark.add_argument("--labels")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--action-budgets", type=int, nargs="+", default=[5, 10, 20])
    benchmark.add_argument("--trials", type=int, default=3)
    benchmark.add_argument("--allow-deterministic", action="store_true")
    benchmark.add_argument("--ablations", action="store_true")
    benchmark.add_argument("--suite", choices=["legacy", "controlled-five"], default="legacy")
    benchmark.add_argument("--max-total-tokens", type=int, default=100000, help="Controlled suite: observed-token stopping ceiling with estimated preflight")
    benchmark.add_argument("--active-context-tokens", type=int, help="Controlled suite: pinned study-token ceiling including transient recovery")
    benchmark.add_argument("--input-tokens", type=int, help="Controlled suite: pinned study-token ceiling for the complete explicit message envelope")
    benchmark.add_argument("--max-llm-cost-usd", type=float, help="Controlled suite: observed cost ceiling; requires configured prices")
    benchmark.add_argument("--require-evidence", action="store_true", help="Controlled suite: require an evidence record or explicit abstention at each decision")
    benchmark.add_argument("--factual-questions", help="Public fixed factual questions only; reserves one common total model call for terminal measurement")
    benchmark.add_argument("--methods", nargs="+", help="Method names in the chosen suite; default uses every method")
    benchmark.add_argument("--trial-seeds", nargs="+", type=int, help="One recorded seed per trial; hosted sampling is not claimed reproducible")
    add_run_options(benchmark)
    judge = commands.add_parser("judgments-import", help="Import adjudicated human relevance CSV")
    judge.add_argument("csv")
    judge.add_argument("--output", required=True)
    judge.add_argument("--scope", choices=["pooled", "exhaustive"], default="pooled")
    claims = commands.add_parser("claim-pool", help="Blind independently assessed factual claims from controlled research runs")
    claims.add_argument("experiment")
    claims.add_argument("--output", required=True)
    research = commands.add_parser("score-research", help="Offline independent relevance/subtopic/claim scoring for controlled-five")
    research.add_argument("experiment")
    research.add_argument("--labels")
    research.add_argument("--subtopics")
    research.add_argument("--claim-judgments")
    evaluate = commands.add_parser("evaluate", help="Score a recorded run using independent judgments")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--query-id", required=True)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument("--output", default="data/evaluation.json")
    pool = commands.add_parser("pool", help="Build a blinded human assessment pool from actual runs")
    pool.add_argument("experiment")
    pool.add_argument("--output", required=True)
    report = commands.add_parser("report", help="Export figures and tables from recorded labeled experiments")
    report.add_argument("experiment")
    report.add_argument("--final", action="store_true", help="Require 10–20 judged questions, four methods and >=3 hosted trials")
    score = commands.add_parser("score-experiment", help="Score saved experiment runs without repeating retrieval")
    score.add_argument("experiment")
    score.add_argument("--labels", required=True)
    export = commands.add_parser("export", help="Export ranked results or persistent event history")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run artifact directory")
    source.add_argument("--events", help="Run ID in the SQLite audit log")
    export.add_argument("--format", choices=["csv", "json"], default="csv")
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # HTTP library info logs can contain credential-bearing query strings from third-party APIs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        if args.db:
            settings.database = args.db
        with Database(settings.database) as db:
            client = OpenAlexClient(db, settings)
            basic = args.command in ("search", "get", "cached")
            logged_session = basic or args.command == "graph"
            try:
                if logged_session:
                    db.event(client.run_id, "session_start", {"command": args.command,
                        "query": getattr(args, "query", None), "paper_id": getattr(args, "paper_id", None),
                        "filter": getattr(args, "filter", None), "limit": getattr(args, "limit", None),
                        "refresh": getattr(args, "refresh", False)})
                if args.command == "graph":
                    graph_command(args, db, client)
                    return
                if not basic and advanced_command(args, db, client, settings):
                    return
                if args.command == "search":
                    papers = client.search(args.query, args.limit, filters=args.filter, refresh=args.refresh)
                    db.event(client.run_id, "search", {"query": args.query, "filter": args.filter,
                                                       "ids": [p.id for p in papers]})
                elif args.command == "get":
                    papers = [client.get(args.paper_id, refresh=args.refresh)]
                else:
                    papers = db.papers()
                if getattr(args, "output", None):
                    export_papers(papers, args.output)
                display(papers, details=args.command == "get")
                print(json.dumps({"run_id": client.run_id, "usage": asdict(client.usage)}))
            finally:
                if logged_session:
                    failure = sys.exc_info()[0]
                    status = "interrupted" if failure is KeyboardInterrupt else "error" if failure else "ok"
                    summary = {"run_id": client.run_id, "status": status, "usage": asdict(client.usage)}
                    try:
                        db.event(client.run_id, "session_end", summary)
                    except sqlite3.Error:
                        logging.getLogger(__name__).warning("Could not persist final session statistics")
                    if failure:
                        print(json.dumps(summary), file=sys.stderr)
                client.close()
    except KeyboardInterrupt:
        print("Interrupted. Retrieval cache and run checkpoints are preserved.", file=sys.stderr)
        raise SystemExit(130) from None
    except (RetrievalError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except sqlite3.Error as exc:
        print(f"Error: database operation failed ({type(exc).__name__}); check --db and write permissions", file=sys.stderr)
        raise SystemExit(1) from None
