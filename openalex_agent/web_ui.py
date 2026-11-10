"""Small Streamlit view of the existing academic retrieval pipeline."""

import json
import os
from pathlib import Path

import streamlit as st

from .config import Settings
from .controller import Budget
from .context_planner import InferenceLimits
from .graph import RELATIONS
from .research import (PROJECT_ROOT, ResearchConfig, corpus_options, graph_view,
                       load_research, paper_view, progress, redact, saved_runs, start_research)

ACTIVE = {"starting", "running"}


def render_activity(state):
    summary = progress(state)
    if state["error"]:
        st.error(state["error"]["message"])
        st.caption("Recorded error category: " + state["error"]["type"])
    elif state["status"] == "completed":
        st.success("Run completed · " + (summary["stop_reason"] or "termination recorded"))
    else:
        st.info("Preparing the final evidence-grounded answer." if state.get("phase") == "synthesis" else
                "Run in progress. Waiting for the next committed execution event.")
    st.write(summary["evidence"])
    counters = [("HTTP attempts", summary["http_attempts"]), ("Successful responses", summary["successful_responses"]),
                ("Cache hits", summary["cache_hits"]), ("API errors", summary["api_errors"]),
                ("Unique records returned", summary["retrieved"]), ("Metadata records admitted", summary["inspected"]),
                ("Returned record observations", summary["returned_record_observations"]),
                ("Sources sent to model", summary["sources_sent_to_model"]),
                ("Actions", summary["actions"]), ("Rounds", summary["rounds"]),
                ("Fallbacks", summary["fallbacks"]), ("Model calls (all stages)", summary["planner_calls"]),
                ("Reported input tokens", summary["input_tokens"]), ("Reported output tokens", summary["output_tokens"]),
                ("Accepted context edits", summary["context_edits"])]
    columns = st.columns(4)
    for i, (label, value) in enumerate(counters):
        columns[i % 4].metric(label, value)
    st.caption("The metadata cap counts unique admitted records, including search batches. Returned observations include "
               "repeat records across actions. Sources sent counts structured excerpts in model inputs; it does not measure "
               "papers read, full-text access, or understanding.")
    if summary["binding_limits"]:
        st.write("Reached limits: " + ", ".join(summary["binding_limits"]))
    if summary["api_errors"]:
        st.warning("OpenAlex errors were recorded. A successful retry does not erase those failures; inspect the trace below.")
    if summary["fallbacks"]:
        st.warning("The controller used fallback after a planner or validation failure. These actions are not accepted LLM decisions.")
    comparison = state["comparison"]
    if comparison.get("performed"):
        st.write(f"Compared {comparison['observed_works']} returned work IDs with the verified "
                 f"{comparison['baseline_papers']}-paper pilot: {comparison['outside_pilot_count']} were absent from that snapshot.")
        st.caption(comparison["meaning"])
        with st.expander("Work IDs absent from the pilot"):
            st.json(comparison["outside_pilot_ids"])
    else:
        st.caption("Pilot comparison pending." if comparison.get("baseline_available") else
                   "Pilot comparison unavailable; no outside-pilot discovery count is claimed.")
    if summary["graph"]:
        st.caption("Latest committed graph counts")
        st.json(summary["graph"], expanded=False)
    decisions = [e for e in state["events"] if e["kind"] == "decision"]
    if decisions:
        st.write("Latest selected action")
        st.json({k: decisions[-1][k] for k in ("action", "arguments", "reason") if k in decisions[-1]})
    st.caption("The trace shows stored execution events. Action explanations are recorded planner/controller text, "
               "not paper metadata, hidden reasoning, or evidence of scholarly relevance.")
    kinds = {"start", "decision", "observation", "invalid_decision", "planner_call", "request", "request_error",
             "cache_hit", "controller_error", "stop_decision", "stop", "unavailable_target", "identifier_redirect",
             "context_model_call", "context_edit", "context_observation", "context_recovery", "evidence_record", "context_budget_stop",
             "live_model_input", "live_model_call", "live_synthesis", "live_planning_stop", "live_synthesis_error"}
    events = [e for e in state["events"] if e["kind"] in kinds]
    for event in events[-50:]:
        title = f"{event['database_seq']} · {event['kind']}"
        if event.get("action"):
            title += " · " + event["action"]
        with st.expander(title):
            st.json({k: v for k, v in event.items() if k not in ("context", "papers")})
            if event["kind"] == "observation":
                st.caption("Committed paper metadata, including new observations and updates")
                for paper in event.get("papers", []):
                    st.text(f"{paper['id']} · {paper.get('title') or 'Title unavailable'}")
    if len(events) > 50:
        st.caption("Showing the latest 50 events; the complete trace is available below.")
    if state["events"]:
        st.download_button("Download complete activity trace", data="".join(json.dumps(e, ensure_ascii=False) + "\n"
            for e in state["events"]), file_name="activity.jsonl", mime="application/x-ndjson")


def render_synthesis(state):
    st.subheader("Research answer")
    result = state.get("result") or {}
    synthesis = result.get("synthesis")
    if not synthesis:
        if state["status"] in ACTIVE:
            st.info("The answer will appear here after research ends, within the selected inference limits.")
        else:
            st.info("This run contains ranked metadata only. Reopening a saved run does not generate an answer or make model calls.")
        return
    st.caption("Research stopped: " + (synthesis.get("stop_reason") or "recorded termination"))
    st.caption(f"{len(synthesis['sources'])} selected sources from {synthesis['metadata_admitted']} admitted records. "
               "Synthesis reloads encountered metadata within the same limits. It is not a benchmark memory score.")
    answer = synthesis.get("answer")
    if answer:
        st.warning("Qualified answer from metadata excerpts. Citation IDs and exact quotes were checked; "
                   "scientific claim support has not been independently reviewed. Full text was not read.")
        for key, title in (("answer", "Direct answer"), ("foundations", "Foundational approaches"), ("comparison", "Comparison")):
            st.write("**" + title + "**")
            if not answer[key]:
                st.write("Not established by the supplied evidence.")
            for claim in answer[key]:
                st.write(claim["text"])
                for citation in claim["citations"]:
                    st.markdown(f"[Source: {citation['work_id'].rsplit('/', 1)[-1]}]({citation['work_id']})")
                    st.text('“' + citation["quote"] + '”')
        st.write("**Limitations and unresolved questions**")
        for limitation in answer["limitations"]:
            st.write(limitation)
        st.download_button("Download research answer", json.dumps(synthesis, indent=2, ensure_ascii=False),
                           file_name="research-answer.json", mime="application/json")
    else:
        st.warning("A model synthesis is unavailable: " + synthesis["reason"] +
                   ". The recorded excerpts below are an evidence digest, not a completed scientific answer.")
    with st.expander("Source evidence", expanded=not bool(answer)):
        for source in synthesis["sources"]:
            st.write(source["title"])
            st.markdown(f"[{source['id']}]({source['id']}) · {source.get('year') or 'Year unavailable'}")
            st.text(source.get("abstract_excerpt") or "Abstract unavailable; title metadata only.")


def render_results(state):
    result = state.get("result")
    if not result:
        st.info("Ranked results appear after the run ends. Committed paper observations are available in Activity.")
        return
    if state["status"] == "failed":
        st.warning("Partial results from a failed retrieval run. They do not establish successful live retrieval.")
    st.caption("Titles, abstracts, authors and citation counts are retrieved metadata. The displayed score is the "
               "existing lexical ranking score, not a relevance judgment or an LLM-generated summary.")
    if not result["ranked"]:
        st.info("The backend returned no ranked papers.")
    for rank, item in enumerate(result["ranked"], 1):
        paper = paper_view(item)
        with st.container(border=True):
            st.text(f"{rank}. {paper['title']}")
            st.write(paper["authors"])
            st.caption(f"{paper['year'] or 'Year unavailable'} · {paper['venue'] or 'Venue unavailable'} · "
                       f"Citations: {paper['citation_count'] if paper['citation_count'] is not None else 'unavailable'}")
            st.write("Ranking score: " + (f"{paper['score']:.4f}" if paper["score"] is not None else "unavailable"))
            left, right = st.columns(2)
            left.link_button("OpenAlex record", paper["openalex_link"])
            if paper["doi_link"]:
                right.link_button("DOI", paper["doi_link"])
            with st.expander("Abstract and recorded inclusion evidence"):
                st.text(paper["abstract"] or "Abstract unavailable")
                st.write("Recorded ranking explanation")
                st.text(paper["ranking_reason"] or "Unavailable")
                st.write("Recorded discovery path")
                st.json(paper["discovery_path"])
                st.caption(f"Provenance: {paper['provenance'] or 'unavailable'} · Retrieved: {paper['retrieved_at'] or 'unavailable'}")
                if paper["limitations"]:
                    st.json(paper["limitations"])
    st.download_button("Download ranked run result", json.dumps(result, indent=2, ensure_ascii=False),
                       file_name="result.json", mime="application/json")


def _draw_graph(view):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openalex-ui-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from((n["id"], n) for n in view["nodes"])
    graph.add_edges_from((e["source"], e["target"]) for e in view["edges"])
    from .ui_theme import NODE_COLORS, palette, method_colors
    theme = palette()
    colors = method_colors()
    node_palette = dict(paper=colors['llm'], author=colors['clm'], topic=colors['llm_graph'], institution=colors['clm_graph'], unresolved=colors['standard'])
    colors = [node_palette['unresolved'] if a.get("type") == "paper" and not a.get("resolved") else node_palette.get(a.get("type"), node_palette['unresolved'])
              for _, a in graph.nodes(data=True)]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor(theme['background']);ax.set_facecolor(theme['background'])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=7), ax=ax, node_color=colors, node_size=250,
                     labels={n: n.rsplit("/", 1)[-1] for n in graph}, font_size=7, arrows=True, arrowsize=10,
                     font_color=theme['text'],edge_color="#8BA58B")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def render_graph(state, job, secrets=()):
    if not job.config.graph_memory:
        st.info("Graph memory is disabled. " + ("The matched research policy retains all six acquisition tools."
                if job.config.controlled else "This run uses the existing search-only iterative method."))
        return
    run_id = state.get("run_id")
    directory = Path(state["directory"]) / "runs" / run_id if run_id else None
    if directory is None or not (directory / "graph.json").is_file():
        st.info("The graph explorer becomes available after the first graph checkpoint.")
        return
    observations = [e for e in state["events"] if e["kind"] == "observation"]
    papers = {p["id"]: p for event in observations for p in event.get("papers", [])}
    if not papers:
        st.info("No inspected papers are available as graph seeds yet.")
        return
    seed = st.selectbox("Neighborhood seed", sorted(papers), key="graph_seed",
                        format_func=lambda p: p.rsplit("/", 1)[-1] + " · " + (papers[p].get("title") or "Unknown title")[:90])
    col1, col2 = st.columns(2)
    hops = col1.slider("Neighborhood hops", 0, 2, 1)
    cap = col2.slider("Visible node limit", 5, 50, 25)
    relations = st.multiselect("Relationships", list(RELATIONS), default=list(RELATIONS))
    if not st.checkbox("Open graph explorer", value=True):
        return
    try:
        view = redact(graph_view(str(directory), seed, max_nodes=cap, hops=hops, relations=relations), secrets)
    except (OSError, ValueError, KeyError):
        st.warning("A validated graph checkpoint is not available yet. Refresh after the next committed action.")
        return
    st.caption("Forest: retrieved paper · sand: unresolved paper · clay: author · olive: topic · eucalyptus: institution. "
               "Arrows follow the stored edge direction. This is a bounded view, not the entire graph.")
    _draw_graph(view)
    nodes = [{"id": n["id"], "type": n.get("type"), "name": n.get("name") or n.get("metadata", {}).get("title"),
              "retrieval_state": ("retrieved" if n.get("resolved") else "unresolved") if n.get("type") == "paper" else None,
              "depth": n.get("depth")} for n in view["nodes"]]
    st.dataframe(nodes, hide_index=True)
    st.dataframe([{k: e.get(k) for k in ("source", "target", "relation", "provenance")} for e in view["edges"]], hide_index=True)
    selected = st.selectbox("Inspect visible node", [n["id"] for n in view["nodes"]])
    with st.expander("Stored node attributes"):
        st.json(next(n for n in view["nodes"] if n["id"] == selected))


def render_context(state, job, secrets):
    run_id = state.get("run_id")
    if run_id:
        requests = sorted((job.directory / "runs" / run_id).glob("live-request-*.json"))
        if requests:
            selected = st.selectbox("Live hosted request (planning and synthesis)", [p.name for p in requests])
            try:
                st.json(redact(json.loads((requests[0].parent / selected).read_text(encoding="utf-8")), secrets), expanded=False)
            except (OSError, ValueError):
                st.info("The request snapshot is being committed; refresh after the next event.")
            st.caption("System instructions, selected evidence and response for each hosted attempt. Synthesis may reload "
                       "encountered metadata; CLM text and archive remain separate below.")
    if not job.config.controlled:
        st.info("Legacy conventional context is selected externally. No model-controlled context edits occurred.")
        return
    if not run_id:
        st.info("Context storage appears after the run initializes.")
        return
    directory = job.directory / "runs" / run_id / "context"
    if not directory.is_dir():
        st.info("No context snapshots are available yet.")
        return
    try:
        if job.config.context_management == "clm":
            st.caption("Model-editable active context. System instructions, original question, scope and budgets remain protected outside this file.")
            st.caption("This is the latest file state; request snapshots show exactly what was sent. A final retrieval may occur after the last planning call.")
            st.code(redact((directory / "active_context.txt").read_text(encoding="utf-8"), secrets), language="text")
        else:
            st.caption("Conventional context uses the current externally selected observation. Archive reads do not constitute CLM edits.")
        requests = sorted(directory.glob("request-*.json"))
        if requests:
            selected = st.selectbox("Actual inference input/output snapshot", [p.name for p in requests])
            st.json(redact(json.loads((directory / selected).read_text(encoding="utf-8")), secrets), expanded=False)
        transitions = sorted(directory.glob("transition-*.json"))
        if transitions:
            selected = st.selectbox("Context transition", [p.name for p in transitions])
            record = redact(json.loads((directory / selected).read_text(encoding="utf-8")), secrets)
            st.write(f"Source: {record['source']} · Accepted: {record['accepted']} · "
                     f"Characters: {record['before_chars']} → {record['after_chars']}")
            st.caption("Environment appends and actual model-requested edits are recorded separately. Retained quotes are not verified claim support.")
            left, right = st.columns(2)
            left.code(record["before"], language="text")
            right.code(record["after"], language="text")
            st.json({k: v for k, v in record.items() if k not in ("before", "after")})
    except (OSError, ValueError, KeyError):
        st.warning("A context snapshot is being committed or cannot be read; refresh after the next event.")


def render_run(job, polling, secrets=()):
    state = job.snapshot()
    if polling and state["status"] not in ACTIVE:
        # Re-enable configuration controls and stop the automatic polling timer.
        st.rerun(scope="app")
    st.caption("Run artifacts: " + state["directory"])
    st.write("Displayed run: " + (state["run_id"] or "initializing") + " · " +
             ("Live OpenAlex" if state["mode"] == "live" else "Fixed corpus"))
    st.text(job.config.question)
    answer, activity, results, graph, context = st.tabs(["Research answer", "Agent activity", "Academic papers", "Knowledge graph", "Context management"])
    with answer:
        render_synthesis(state)
    with activity:
        render_activity(state)
    with results:
        render_results(state)
    with graph:
        render_graph(state, job, secrets)
    with context:
        render_context(state, job, secrets)


def main():
    st.set_page_config(page_title="OpenAlex Research Studio", page_icon=":material/menu_book:", layout="wide")
    from .ui_theme import apply_theme
    apply_theme()
    st.markdown('<div class="oa-eyebrow">OpenAlex / Research Systems</div>',unsafe_allow_html=True)
    try:
        settings = Settings.from_env()
    except ValueError:
        st.error("Server configuration is invalid. Check .env numeric settings; credentials are never entered in this page.")
        return
    st.sidebar.title("OpenAlex Agent")
    st.sidebar.caption("Literature discovery & scientific evaluation")
    workspaces = ["Scientific Evaluation", "Individual research", "Experiment Dashboard", "Long-Horizon Testing", "Saved Experiments"]
    default_workspace = os.environ.get("OPENALEX_UI_WORKSPACE", "Scientific Evaluation")
    page = st.sidebar.radio("Workspace", workspaces,
        index=workspaces.index(default_workspace) if default_workspace in workspaces else 0)
    st.sidebar.divider()
    st.sidebar.caption("PRIMARY STUDY")
    st.sidebar.caption("Standard · LLM · LLM + graph · CLM · CLM + graph")
    st.sidebar.caption("Official Meta CLM is exploratory.")
    if page=="Scientific Evaluation":
        from .evaluation_ui import render_evaluation
        render_evaluation(PROJECT_ROOT,settings)
        return
    if page!="Individual research":
        from .dashboard_ui import main as dashboard_main
        dashboard_main(page,settings,PROJECT_ROOT)
        return
    st.title("Individual research")
    st.caption("Discover literature, inspect source metadata, and follow the recorded research process.")
    job = st.session_state.get("research_job")
    running = bool(job and job.snapshot()["status"] in ACTIVE)
    if job and not running:
        render_run(job, False, tuple(v for v in (settings.api_key, settings.llm_api_key, settings.email) if v))
    with st.expander("Reopen a saved UI run"):
        history = saved_runs(PROJECT_ROOT)
        if history:
            previous = st.selectbox("Saved run", history, disabled=running)
            if st.button("Open saved run", disabled=running):
                try:
                    st.session_state.research_job = load_research(previous, root=PROJECT_ROOT)
                except (OSError, ValueError, KeyError, TypeError):
                    st.error("The selected saved run could not be read. Its files were not modified.")
                else:
                    st.rerun()
        else:
            st.caption("Ended UI runs appear here. Opening one does not make API calls.")
    st.subheader("Research question and retrieval settings")
    question = st.text_area("Research question", max_chars=1000, key="question", disabled=running,
                            placeholder="How does quantum error correction improve fault tolerance in superconducting qubits?")
    left, right = st.columns(2)
    mode_label = left.selectbox("Retrieval source", ["Live OpenAlex", "Cached corpus / snapshot"], disabled=running)
    mode = "live" if mode_label == "Live OpenAlex" else "snapshot"
    secrets = tuple(v for v in (settings.api_key, settings.llm_api_key, settings.email) if v)
    model = redact(settings.llm_model, secrets) or "not configured"
    planner_label = right.selectbox("Planner configuration", [f"Hosted · {model} (.env)", "Deterministic (no LLM calls)"], disabled=running)
    planner = "hosted" if planner_label.startswith("Hosted") else "deterministic"
    st.caption("Terra is the default hosted profile. Qwen/Groq or Gemini can use the same server-side .env configuration. "
               "No provider keys are sent to this page.")
    corpus = None
    if mode == "snapshot":
        choices = corpus_options(PROJECT_ROOT)
        choice = st.selectbox("Existing corpus", choices + ["Other local snapshot"], disabled=running)
        corpus = st.text_input("Corpus directory", disabled=running) if choice == "Other local snapshot" else choice
        st.info("Fixed corpus selected. Live OpenAlex retrieval is unavailable; every action stays within this snapshot.")
    else:
        st.info("Live OpenAlex selected. Each run uses a fresh database; cache reuse within the run is counted separately. "
                "HTTP success is verified from actual events, not from the selected mode.")
    graph_memory = st.checkbox("Enable graph memory", value=True, disabled=running)
    policy = st.selectbox("Context and experiment policy", ["Conventional (existing policy)",
        "Conventional (matched six tools)", "CLM-inspired (matched six tools)"], disabled=running)
    controlled = policy != "Conventional (existing policy)"
    context_management = "clm" if policy.startswith("CLM") else "conventional"
    if controlled:
        st.caption("Controlled research policy: all six acquisition tools and common metadata eligibility are available with or without graph memory. "
                   "CLM uses bounded JSON text edits, not the paper's unrestricted Bash implementation. Deterministic mode is a scripted engineering demo.")
    if not graph_memory and not controlled:
        st.caption("Uses the existing graph-free iterative policy: SEARCH_PAPERS and STOP_RETRIEVAL. "
                   "Graph mode retains all six existing actions. This toggle is not the controlled pilot ablation.")
    a, b, c = st.columns(3)
    actions = int(a.number_input("Maximum actions", 1, 100, 5, disabled=running))
    inspected = int(b.number_input("Maximum admitted metadata records", 1, 10000, 30, disabled=running,
        help="Search and expansion batches consume this cap. It does not count papers read by the model."))
    api_cap = max(1, min(500, settings.max_requests))
    requests = int(c.number_input("Maximum HTTP attempts", 1, api_cap, min(10, api_cap), disabled=running,
                                  help="Includes retries. The configured .env ceiling also applies."))
    with st.expander("Additional existing limits"):
        rounds = int(st.number_input("Maximum rounds", 1, 100, 5, disabled=running))
        calls = int(st.number_input("Maximum LLM calls", 1, 100, 4, disabled=running,
            help="Shared by planning, context edits, recovery and final synthesis."))
        page = int(st.number_input("Papers per action", 1, 100, 5, disabled=running))
        output_k = int(st.number_input("Ranked output size", 1, 200, 20, disabled=running))
        patience = int(st.number_input("Stagnant rounds before stopping", 1, 100, 3, disabled=running))
        context = int(st.number_input("Planner context character limit", 4000, 64000, 16000, step=1000, disabled=running))
        tokens = int(st.number_input("Research token ceiling", 1000, 10000000, 100000, disabled=running,
                                     help="All hosted stages: observed usage plus conservative preflight estimate, not a provider-exact prepaid cap."))
        dollars = st.number_input("Research hosted dollar ceiling (0 = unset)", 0.0, 1000.0, 0.0, disabled=running,
                                  help="All hosted stages. Requires positive input/output prices; preflight reserves maximum completion cost using configured prices.")
    synthesis = st.checkbox("Produce a final evidence-grounded answer", value=True, disabled=running)
    if planner == "deterministic":
        st.caption("Deterministic mode returns an evidence digest without model calls. It does not generate a scientific synthesis.")
    planning_allowance = max(0, calls - int(synthesis))
    st.caption(f"Effective retrieval action ceiling: {min(actions, rounds, inspected)} (actions, rounds and metadata caps all apply). "
               f"Each action admits at most {page} records. Hosted planning, edits and recovery share {planning_allowance} calls; "
               f"{int(synthesis)} call is reserved for the final answer within the {calls}-call ceiling. "
               "Token and dollar limits may further reduce this allowance. Limits are never raised automatically.")
    if rounds < actions:
        st.warning(f"The {rounds}-round ceiling will stop research before {actions} actions. Each acquisition consumes one round.")
    hosted_missing = planner == "hosted" and not all((settings.llm_base_url, settings.llm_api_key, settings.llm_model))
    if hosted_missing:
        st.warning("Hosted planner configuration is incomplete. Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL on the server, or select deterministic.")
    paid=st.checkbox("Confirm paid hosted retrieval",value=False,disabled=running or planner!="hosted")
    if planner=="hosted":st.caption(f"At most {calls} total model calls, including synthesis, and {requests if mode=='live' else 0} live OpenAlex attempts within the stated limits. This individual demonstration is not an approved empirical comparison.")
    if st.button("Start retrieval", type="primary", disabled=running or hosted_missing or planner=="hosted" and not paid):
        try:
            from .experiment_dashboard import reserved_question
            if reserved_question(question,PROJECT_ROOT):
                raise ValueError("Reserved final questions require the approved study workflow")
            config = ResearchConfig(question, mode=mode, planner=planner, graph_memory=graph_memory, corpus=corpus,
                budget=Budget(max_actions=actions, max_rounds=rounds, max_inspected=inspected,
                    max_api_requests=requests, max_llm_calls=calls, page_size=page, output_k=output_k,
                    patience=patience, max_context_chars=context), controlled=controlled, context_management=context_management,
                inference_limits=InferenceLimits(tokens, dollars if dollars else None), final_synthesis=synthesis)
            st.session_state.research_job = start_research(config, settings, root=PROJECT_ROOT)
        except ValueError as exc:
            st.error("Cannot start: " + redact(str(exc), secrets))
        else:
            st.rerun()
    if job and running:
        st.fragment(run_every=1.0 if running else None)(render_run)(job, running, secrets)
    elif not job:
        st.caption("Start a run to inspect its actual activity, ranked papers and knowledge graph.")
