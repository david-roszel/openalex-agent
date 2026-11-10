"""Simple submission interface; all research runs use research.start_research."""
from dataclasses import replace
import json
import os
from urllib.parse import quote

import streamlit as st

from .live_profiles import load_live_settings, thorough_config
from .live_research import answer_markdown
from .research import PROJECT_ROOT, start_research, saved_runs, load_research, progress

ARCHITECTURE_LABELS = {"combined": "LLM + citation graph + CLM", "conventional": "Conventional LLM agent",
    "graph": "LLM + citation graph", "clm": "LLM + CLM", "clm_graph": "LLM + citation graph + CLM",
    "llm_graph": "LLM + citation graph", "llm": "Conventional LLM agent"}


def citation_links(claim):
    # Several exact supporting quotations can belong to the same publication.
    # Deduplicate display links only; retain every original evidence receipt.
    ids = dict.fromkeys(c["work_id"] for c in claim["citations"])
    return " ".join(f"[{pid.rsplit('/', 1)[-1]}]({pid})" for pid in ids)


def saved_run_label(identifier):
    try:
        state = json.loads((PROJECT_ROOT / "data/ui-runs" / identifier / "summary.json").read_text())
        result = state.get("result") or {}
        question = (result.get("question") or "Saved research").strip()
        architecture = ARCHITECTURE_LABELS.get(result.get("architecture"), result.get("architecture", "configuration not recorded"))
        return f"{question[:72]} · {architecture} · {identifier[:8]}"
    except (OSError, ValueError, TypeError, AttributeError):
        return identifier + " · saved record unavailable"


def render_job(job, *, polling=False):
    state = job.snapshot()
    active = state["status"] in ("starting", "running")
    if polling and not active:
        st.rerun(scope="app")
    result = state.get("result") or {}
    p = progress(state)
    if result:
        st.caption("Recorded research result · no research is running" if not active else "Current research question")
        st.text(result.get("question", job.config.question).strip())
        architecture = result.get("architecture") or ("combined" if job.config.graph_memory and job.config.context_management == "clm"
            else "graph" if job.config.graph_memory else "clm" if job.config.context_management == "clm" else "conventional")
        st.caption("Recorded architecture: " + ARCHITECTURE_LABELS.get(architecture, architecture)
            + f" · {job.config.max_seconds:g}-second ceiling · hosted ceiling "
            + (f"${job.config.inference_limits.max_llm_cost_usd:.2f}" if job.config.inference_limits.max_llm_cost_usd is not None else "not configured"))
        st.caption("Run: " + str(state.get("run_id") or job.directory.name))
        if not active and result.get("stop_reason"):
            st.caption("Recorded stop: " + result["stop_reason"].replace("_", " "))
    if active:
        stage = state.get("phase", "research")
        st.status("Synthesizing the research answer…" if stage == "synthesis" else "Investigating the literature…", expanded=False)
        st.caption(f"{p['actions']} research actions · {p['inspected']} metadata records · {p['sources_sent_to_model'] or 0} sources sent to the model")
    if state.get("error"):
        st.error(state["error"]["message"])
    synthesis = result.get("synthesis") or {}
    answer = synthesis.get("answer")
    if answer:
        st.subheader("Research answer")
        if synthesis.get("question_completion") == "incomplete_evidence" or result.get("research_coverage", {}).get("adequate") is False:
            st.warning("Partial answer: the investigation did not establish adequate evidence for the complete question. Citations are traceable, but the requested comparison or other components may remain unresolved.")
        for field, label in (("answer", ""), ("foundations", "Research background"),
                ("comparison", "Comparison and findings"), ("conclusion", "Conclusion")):
            if label and answer.get(field):
                st.markdown("### " + label)
            for claim in answer.get(field, []):
                references = citation_links(claim)
                st.markdown(claim["text"] + " " + references)
        st.markdown("### Limitations and unresolved questions")
        for text in answer["limitations"]:
            st.write(text)
        st.caption("Based on retrieved abstracts and metadata, not full-text reading. Citation and quote traceability was checked; scientific claim support has not been independently adjudicated.")
        st.download_button("Download answer (.md)", answer_markdown(result), "research-answer.md", "text/markdown")
    elif result:
        st.warning("Research collected evidence, but a complete supported synthesis was not produced: " + str(synthesis.get("reason", result.get("stop_reason"))))
    if synthesis.get("sources"):
        cited = {c["work_id"] for key in ("answer", "foundations", "comparison", "conclusion")
                 for claim in (answer or {}).get(key, []) for c in claim["citations"]}
        st.subheader("Supporting publications")
        for source in synthesis["sources"]:
            if cited and source["id"] not in cited:
                continue
            with st.expander(f"{source['title']} ({source.get('year') or 'year unavailable'})"):
                st.write(", ".join(source.get("authors", [])) or "Authors unavailable")
                st.markdown(f"[OpenAlex record]({source['id']})")
                if source.get("doi"):
                    doi = source["doi"].removeprefix("https://doi.org/").removeprefix("http://doi.org/")
                    if doi.startswith("10."):
                        st.markdown(f"[DOI: {doi}](https://doi.org/{quote(doi, safe='/')})")
                    else:
                        st.text("DOI: " + source["doi"])
                st.text(source.get("abstract_excerpt") or "Abstract unavailable")
    elif result.get("ranked"):
        st.subheader("Retrieved publications")
        for item in result["ranked"][:15]:
            paper = item["paper"]
            st.markdown(f"[{paper.get('title') or paper['id']}]({paper['id']})")
    with st.expander("Research trace and resource use"):
        st.json({"progress": p, "stop_reason": result.get("stop_reason"), "coverage": result.get("research_coverage"),
            "mechanisms": result.get("mechanisms"), "spending": result.get("spending"), "artifacts": str(job.directory)})
        events = state["events"]
        st.caption("Model-authored edits are context_edit events; environment observation updates are context_observation events. live_model_input records show directed edges actually sent, not proof of model reliance or improved research quality.")
        for event in (events[-40:] if active else events):
            if event["kind"] in ("decision", "research_coverage", "live_stop_reconsidered", "invalid_decision", "live_model_input", "context_edit", "context_observation", "context_recovery", "controller_error"):
                st.json({k: v for k, v in event.items() if k not in ("papers", "before", "after", "requested_after")}, expanded=False)
        st.download_button("Download complete trace", "\n".join(json.dumps(e, ensure_ascii=False) for e in events), "research-trace.jsonl", "application/x-ndjson")


def main():
    st.set_page_config(page_title="OpenAlex Live Research", page_icon="📚", layout="centered")
    st.title("OpenAlex Live Research")
    st.caption("Ask an academic question. Follow the evidence. Read a source-grounded research answer.")
    read_only = os.getenv("OPENALEX_DEMO_READ_ONLY", "").lower() in ("1", "true", "yes")
    if read_only:
        st.info("Read-only demonstration: open a saved research result below. Research is disabled; no retrieval or model requests can be started from this page.")
    try:
        settings = load_live_settings()
    except ValueError:
        settings = None
        st.error("Server configuration is invalid. Check .env numeric settings. Saved research can still be opened without starting inference.")
    job = st.session_state.get("research_job")
    running = bool(job and job.snapshot()["status"] in ("starting", "running"))
    question = st.text_area("Research question", max_chars=1000, disabled=running,
        placeholder="How do iterative retrieval and graph-based retrieval differ for multi-hop question answering?")
    with st.expander("Advanced Settings"):
        st.caption("Thorough Research: six-record batches; graph + CLM by default; bounded investigation with synthesis and one content-repair opportunity.")
        variant = st.selectbox("Research configuration", ["combined", "conventional", "graph", "clm"],
            format_func=lambda v: ARCHITECTURE_LABELS[v], disabled=running)
        st.caption("Citation graph adds directed scholarly relationships to planner inputs. CLM lets the model edit its working context. Availability does not establish improved answer quality.")
        st.caption("Default ceilings: 18 research actions, 40 OpenAlex requests, 144 metadata records and 28 total model calls, including context operations and final synthesis. Records retrieved and sources sent to the model are counted separately.")
        planner = st.selectbox("Execution", ["hosted", "deterministic"], disabled=running,
            help="Deterministic is an engineering demonstration and produces an evidence digest, not a model research answer.")
        source = st.selectbox("Source", ["Live OpenAlex", "Local snapshot"], disabled=running)
        corpus = st.text_input("Snapshot directory", disabled=running or source == "Live OpenAlex")
        ceiling = st.number_input("Hosted spending ceiling (USD)", min_value=0.10, max_value=20.0, value=1.50, step=0.10, disabled=running)
        seconds = st.number_input("Time ceiling (seconds)", min_value=60, max_value=1800, value=600, step=60, disabled=running)
        st.caption("Provider credentials are read only from server configuration. This app has no dependency on the V3 corpus or its spending authorization.")
        previous = saved_runs(PROJECT_ROOT)
        if previous:
            selected = st.selectbox("Saved research", previous, format_func=saved_run_label, disabled=running)
            if st.button("Open saved research", disabled=running):
                try:
                    st.session_state.research_job = load_research(selected, root=PROJECT_ROOT)
                except (ValueError, OSError, KeyError, TypeError, AttributeError):
                    st.error("The saved run could not be read; no inference was started.")
                else:
                    st.rerun()
    if planner == "hosted":
        st.caption(f"{'Inactive new-run defaults' if read_only else 'Thorough Research'} · up to ${ceiling:.2f} at configured conservative prices · {seconds // 60} minute limit."
            + ("" if read_only else " Clicking Research authorizes this individual run only."))
    missing = [name for name, value in (("LLM_BASE_URL", settings.llm_base_url), ("LLM_API_KEY", settings.llm_api_key),
        ("LLM_MODEL", settings.llm_model)) if not value.strip()] if settings is not None and planner == "hosted" else []
    if missing and not read_only:
        st.warning("Hosted research is unavailable: configure " + ", ".join(missing) + ". Saved results remain available.")
    if st.button("Research", type="primary", disabled=running or read_only or settings is None or bool(missing)):
        try:
            config = thorough_config(question, variant=variant, planner=planner,
                mode="live" if source == "Live OpenAlex" else "snapshot", corpus=corpus if source != "Live OpenAlex" else None,
                max_seconds=float(seconds))
            if planner == "hosted":
                config = replace(config, inference_limits=replace(config.inference_limits, max_llm_cost_usd=float(ceiling)))
            st.session_state.research_job = start_research(config, settings, root=PROJECT_ROOT)
        except (ValueError, OSError) as exc:
            from .live_safety import scrub
            st.error(scrub(str(exc), (settings.api_key, settings.llm_api_key, settings.email)))
        else:
            st.rerun()
    if job:
        if running:
            st.fragment(run_every=1.0)(render_job)(job, polling=True)
        else:
            render_job(job)


if __name__ == "__main__":
    main()
