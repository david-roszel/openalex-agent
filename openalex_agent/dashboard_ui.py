"""Streamlit views for the shared experiment adapter and recorded artifacts."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import uuid
import hashlib

import pandas as pd
import streamlit as st
from .ui_theme import METHOD_COLORS, method_colors, chart_style

from .context_planner import InferenceLimits
from .controller import Budget
from .experiment_dashboard import (DashboardConfig, CATEGORIES, KINDS, LABELS, QUALITY, OPERATIONAL,
    allowed_path, approval_digest, context_records, equivalence, export_summary, figure_export,
    profile_digest, read_comparison, readiness, saved_experiments, source_digest, start_dashboard, trace_records)
from .experiment_dashboard import assess_saved_copy
from .research import corpus_options, redact

ACTIVE={"starting","running","cancellation_requested"}


@st.cache_data(ttl=10,show_spinner=False)
def catalog(root):
    root=Path(root)
    corpora=corpus_options(root)
    for p in (root/"data").glob("*/corpus/manifest.json"):
        if (p.parent/"papers.jsonl").is_file():corpora.append(str(p.parent.relative_to(root)))
    files=[]
    for parent in (root/"benchmarks",root/"data/study",root/"data/prepilot-development-v1"):
        if parent.exists():files.extend(str(p.relative_to(root)) for p in parent.glob("*.json"))
    return sorted(set(corpora)),sorted(set(files)),saved_experiments(root)


def pick_file(label,files,*,key,disabled=False):
    value=st.selectbox(label,["Not selected",*files],key=key,disabled=disabled)
    return None if value=="Not selected" else value


def _table(rows):
    frame=pd.DataFrame(rows)
    if frame.empty:return frame
    for column in frame:
        frame[column]=frame[column].map(lambda v:"NOT EVALUATED" if v is None or isinstance(v,float) and not math.isfinite(v) else v)
        if frame[column].map(lambda v:isinstance(v,str) and v=="NOT EVALUATED").any():frame[column]=frame[column].map(str)
    return frame


def context_compare(directories,root,secrets,prefix):
    if not directories:
        st.info("No recorded per-run context is available.");return
    chosen=st.multiselect("Context trajectories side by side",list(directories),default=list(directories)[:2],key=prefix+"contexts")
    for name,col in zip(chosen,st.columns(max(1,len(chosen)))):
        with col:
            records=context_records(directories[name],root)
            st.write(name)
            if not records["requests"]:st.info("No observable inference snapshots.");continue
            series=pd.DataFrame([{k:r[k] for k in ("call","active_chars","active_tokens","full_input_tokens")} for r in records["requests"]]).set_index("call")
            st.caption("Actual request sizes. Missing token counts are not reconstructed as provider usage.")
            st.line_chart(series[["active_chars"]].reset_index(),x="call",y="active_chars",color=method_colors().get(name.split(' · ')[0],'#244B3B'))
            if series.active_tokens.notna().any():st.line_chart(series[["active_tokens","full_input_tokens"]].reset_index(),x="call",y=["active_tokens","full_input_tokens"],color=[method_colors()['llm'],method_colors()['clm']])
            st.dataframe([{k:r[k] for k in ("call","recorded_quotes_in_active","recorded_quotes_in_recovery")} for r in records["requests"]],hide_index=True)
            st.caption("Exact text from recorded output quotations only; presence does not establish semantic support or hidden model knowledge.")
            call=st.select_slider("Recorded call",options=[r["call"] for r in records["requests"]],key=prefix+"context-call-"+name)
            request=next(r for r in records["requests"] if r["call"]==call)
            st.code(redact(request["active"],secrets),language="text")
            if request["recovery"]:st.json(redact(request["recovery"],secrets))
            with st.expander("Actual system, input and parsed output"):
                st.json(redact(request["request"],secrets),expanded=False)
            with st.expander("Requested / committed / failed context operations"):
                st.json(redact(records["intents"],secrets),expanded=False)
                for t in records["transitions"]:
                    st.write(f"{t['index']} · {t['source']} · {t.get('status','legacy acceptance='+str(t.get('accepted')))}")
                    st.json(redact(t,secrets),expanded=False)
    st.caption("These are observable editable payloads, intents and committed transitions—not hidden reasoning or authoritative OpenAlex/graph state.")


def render_saved(directory,root,secrets=()):
    try:loaded=read_comparison(directory,root)
    except (ValueError,OSError):st.error("This artifact is outside approved locations or cannot be read.");return
    st.subheader("Recorded experiment · "+loaded["category"])
    prefix=hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    st.caption(loaded["directory"])
    if loaded["category"]=="Diagnostics":st.info("DIAGNOSTIC ONLY — synthetic/scripted correctness evidence, not empirical model superiority.")
    elif loaded["category"]=="UNVERIFIED SAVED EXPERIMENT":st.warning("No dashboard scientific approval is recorded. Raw evidence is inspectable; empirical quality scores are NOT EVALUATED.")
    if loaded["status"]:st.json(redact(loaded["status"],secrets),expanded=False)
    results,contexts,graphs,raw=st.tabs(["Side-by-side results","Context evolution","Actions and graph behavior","Manifests and raw evidence"])
    with results:
        rows=loaded["rows"]
        if rows:
            qids=sorted({r["query_id"] for r in rows})
            q=st.selectbox("Inspect research question",["All questions",*qids],key=prefix+"question")
            trials=sorted({int(r["trial"]) for r in rows})
            trial=st.selectbox("Inspect trial",["All trials",*trials],key=prefix+"trial")
            selected=[r for r in rows if (q=="All questions" or r["query_id"]==q) and (trial=="All trials" or r["trial"]==trial)]
            display=[]
            for r in selected:
                row={k:r.get(k) for k in ("query_id","method","variant","trial","seed","action_budget","evaluation_status")}
                row.update({label:r.get(column) for label,column in {**QUALITY,**OPERATIONAL}.items()});display.append(row)
            st.dataframe(_table(redact(display,secrets)),hide_index=True)
            metric=st.selectbox("Compare available metric",list({**QUALITY,**OPERATIONAL}),key=prefix+"metric")
            column={**QUALITY,**OPERATIONAL}[metric]
            chart=pd.DataFrame(selected)
            if column in chart and pd.to_numeric(chart[column],errors="coerce").notna().any():
                values=chart.assign(value=pd.to_numeric(chart[column],errors="coerce")).dropna(subset=["value"])
                bars=values.groupby(["method","variant","action_budget"]).value.mean().reset_index()
                bars["configuration"]=bars.apply(lambda r:f"{r['method']}/{r['variant']}/budget={r['action_budget']}",axis=1)
                import altair as alt
                plot=alt.Chart(bars).mark_bar(cornerRadiusEnd=3).encode(x=alt.X('configuration:N',title=None),
                    y=alt.Y('value:Q',title=metric),color=alt.Color('method:N',
                    scale=alt.Scale(domain=list(method_colors()),range=list(method_colors().values()))),
                    tooltip=['configuration:N','value:Q'])
                st.altair_chart(chart_style(plot),width='stretch')
                st.caption("Descriptive means over selected recorded questions/trials. Coverage, failures and raw rows remain visible; no significance claim.")
                with st.expander("Existing paired contrasts and graph × CLM interaction"):
                    if loaded["manifest"].get("status")=="complete":
                        from .experiment_integrity import validate_schedule
                        from .research_metrics import paired_comparisons
                        try:
                            validate_schedule(loaded["manifest"],loaded["rows"])
                            paired=paired_comparisons(selected,column)
                            st.json(paired,expanded=False)
                            st.download_button("Export paired descriptive comparisons",json.dumps(paired,indent=2),
                                file_name="paired-comparisons.json",mime="application/json",key=prefix+"paired")
                        except (ValueError,KeyError,TypeError):st.info("Paired analysis unavailable: incomplete trial identity or unmatched repetitions.")
                    else:st.info("Paired analysis waits for a complete recorded schedule.")
                st.download_button("Export comparison figure (PNG)",figure_export(redact({**loaded,"rows":selected},secrets),column),file_name="comparison.png",mime="image/png",key=prefix+"png")
            else:st.info(metric+": NOT EVALUATED")
            st.download_button("Export summary with provenance (CSV)",redact(export_summary({**loaded,"rows":selected}),secrets),file_name="comparison-summary.csv",mime="text/csv",key=prefix+"csv")
        elif not loaded["pressure"]:st.info("No completed benchmark cells yet. Partial state and errors are retained.")
        if loaded["pressure"]:
            valid=[r for r in loaded["pressure"] if r["pressure_valid"]]
            invalid=[r for r in loaded["pressure"] if not r["pressure_valid"]]
            if invalid:st.error("INVALID PRESSURE CONDITION — invalid rows are excluded from aggregate displays.")
            pressure_fields=["architecture","mode","seed","pressure_valid","pressure_ratio","unique_source_pressure_ratio",
                "total_observation_tokens_encountered","unique_delivered_source_excerpt_tokens","active_context_capacity_tokens",
                "max_active_context_tokens","max_full_input_tokens","evidence_retained_in_final_active","current_evidence_retained",
                "evidence_removed_from_active","evidence_omitted_from_selected_view","evidence_recoverable_externally",
                "evidence_exists_in_audit_archive","evidence_successfully_recovered","exact_recovery_accuracy","factual_errors","model_calls","management_calls","management_study_input_tokens",
                "management_study_output_tokens","management_seconds","context_adapter_local_seconds","source_artifact"]
            pressure_fields.insert(-1,"stop_reason")
            st.dataframe(_table([{k:r.get(k) for k in pressure_fields} for r in loaded["pressure"]]),hide_index=True)
            if valid:
                bars=pd.DataFrame(valid).groupby("architecture")[["total_observation_tokens_encountered","max_active_context_tokens"]].mean().reset_index()
                st.bar_chart(bars,x="architecture",y=["total_observation_tokens_encountered","max_active_context_tokens"],color=['#657A3F','#A7794C'])
                st.caption("Only validated diagnostic conditions appear in this chart. Exact recovery concerns the initial probe; unattempted recovery is NOT EVALUATED and correction accuracy is separate.")
            st.download_button("Export pressure records and source paths",json.dumps(redact(loaded["pressure"],secrets),indent=2),file_name="pressure-records.json",mime="application/json",key=prefix+"pressure")
        if loaded["diagnostics"]:st.json(redact(loaded["diagnostics"],secrets))
        if loaded["graph_diagnostics"]:st.json(redact(loaded["graph_diagnostics"],secrets),expanded=False)
    dirs={f"{r['method']} · {r['query_id']} · trial {r['trial']} · {r['variant']} · budget {r['action_budget']}":r["artifact_directory"] for r in loaded["rows"] if r.get("artifact_directory")}
    for i,r in enumerate(loaded["pressure"]):
        try:dirs[f"{r.get('architecture',r['mode'])} · seed {r['seed']} · pressure {i}"]=str(allowed_path(r["directory"],root))
        except ValueError:continue
    for directory in loaded.get("partial_runs",[]):dirs["Incomplete run · "+Path(directory).name]=directory
    with contexts:context_compare(dirs,root,secrets,prefix)
    with graphs:
        selected=st.multiselect("Inspect method execution traces",list(dirs),default=list(dirs)[:2],key=prefix+"traces")
        for name,col in zip(selected,st.columns(max(1,len(selected)))):
            with col:
                trace=trace_records(dirs[name],root);st.write(name)
                st.metric("Repeated action signatures",trace["redundant_actions"])
                decisions=[e for e in trace["events"] if e.get("kind")=="decision"]
                st.dataframe(redact([{k:e.get(k) for k in ("event_seq","action","arguments","reason")} for e in decisions],secrets),hide_index=True)
                st.caption("Citation expansions and discovered IDs are recorded actions/observations. Relevance per action is unavailable without judgments.")
                st.json(redact(trace["result"].get("inspected_ids",[]),secrets),expanded=False)
                with st.expander("Graph-derived planner information"):
                    st.json(redact(trace["graph_planner_inputs"],secrets),expanded=False)
                with st.expander("Final lexical rankings / discovery paths"):
                    st.json(redact(trace["result"].get("ranked",[]),secrets),expanded=False)
                with st.expander("Complete recorded execution trace"):
                    st.json(redact(trace["events"],secrets),expanded=False)
        st.caption("Graph scores affect planning; every current controlled method uses the same final lexical ranker. No graph-ranking effect is claimed.")
    with raw:
        st.json(redact(loaded["dashboard"],secrets),expanded=False)
        st.json(redact(loaded["manifest"],secrets),expanded=False)
        st.download_button("Export manifest and evaluation status",json.dumps(redact({k:v for k,v in loaded.items() if k not in ("rows","pressure")},secrets),indent=2),file_name="experiment-provenance.json",mime="application/json",key=prefix+"manifest")


def render_progress(job,root,secrets):
    state=job.snapshot()
    st.write("Execution status: "+state["status"])
    st.caption(state["directory"])
    a,b=st.columns(2);a.metric("Completed benchmark cells",state["completed_cells"]);b.metric("Committed events observed",state["recorded_events"])
    st.metric("Recorded inference snapshots",state["recorded_inference_snapshots"])
    if state.get("last_event"):st.caption("Latest actual event: "+state["last_event"])
    if state.get("global_usage"):st.json(redact(state["global_usage"],secrets),expanded=False)
    if state["status"] in ACTIVE:
        if st.button("Cancel experiment",key="cancel-experiment"):
            job.cancel();st.rerun()
        st.info("Polling actual committed artifacts. Cancellation is cooperative; an in-flight hosted request can finish before stopping. Offline pressure cancellation occurs between arms.")
    if state.get("error"):st.error(state["error"]["message"]+" ("+state["error"]["type"]+")")
    render_saved(job.directory,root,secrets)


def main(page,settings,root):
    st.title("Research Experiment Dashboard" if page!="Long-Horizon Testing" else "Long-Horizon Testing")
    st.caption("Existing CLI/backend methodology · actual recorded artifacts · independent judgments remain pending")
    secrets=tuple(v for v in (settings.api_key,settings.llm_api_key,settings.email) if v)
    corpora,files,saved=catalog(str(root))
    if page=="Saved Experiments":
        if not saved:st.info("No saved experiment artifacts found.");return
        choices=st.multiselect("Saved experiments to compare",saved,default=saved[:1])
        for i,path in enumerate(choices):
            # Scope widget keys by artifact to compare several experiments safely.
            with st.container(key="saved-experiment-"+str(i)):
                render_saved(allowed_path(path,root),root,secrets)
        if choices:
            with st.expander("Evaluate a fresh copy with independent judgments"):
                st.caption("The existing scorer writes only a new assessed copy. Historical/frozen sources remain unchanged. Empty or incomplete judgment files cannot certify truth.")
                selected=st.selectbox("Experiment to assess",choices)
                relevance=st.text_input("Relevance judgment JSON under approved data locations")
                claim_file=st.text_input("Independent claim-judgment CSV under approved data locations")
                claim_review=st.text_input("Coordinator claim-review attestation JSON (empirical scoring)")
                if st.button("Score new assessed copy"):
                    try:
                        output=assess_saved_copy(selected,root,labels=relevance or None,claims=claim_file or None,claim_review=claim_review or None)
                        st.success("Assessment copy saved: "+str(output));catalog.clear()
                    except (ValueError,OSError,KeyError,TypeError):st.error("Assessment was not completed. Check approval, complete independent judgments and approved paths; source artifacts were not changed.")
        return
    job=st.session_state.get("dashboard_job")
    running=bool(job and job.snapshot()["status"] in ACTIVE)
    if job and st.button("Prepare another experiment",disabled=running):
        st.session_state.dashboard_submission=str(uuid.uuid4())
        del st.session_state["dashboard_job"]
        st.rerun()
    if page=="Long-Horizon Testing":category="Diagnostics";st.info("Scientific category: Diagnostics · offline scripted fixtures")
    else:category=st.selectbox("Scientific category",CATEGORIES,disabled=running)
    kind=st.selectbox("Experiment type",["Long-horizon pressure"] if page=="Long-Horizon Testing" else KINDS,disabled=running)
    defaults=("llm","clm") if kind=="Long-horizon pressure" else ("llm","llm_graph") if kind=="Graph ablation" else ("llm","clm") if kind=="CLM ablation" else tuple(LABELS)
    selected=st.multiselect("Architectures",list(LABELS),default=list(defaults),format_func=LABELS.get,disabled=running,key="architectures-"+kind)
    planner=st.selectbox("Experiment planner",["deterministic"] if category=="Diagnostics" else ["hosted","deterministic"],disabled=running)
    model=st.text_input("Hosted planner model",value=settings.llm_model,disabled=running or planner!="hosted")
    if category=="Diagnostics":st.info("Offline fixtures only. Comparisons use a synthetic G1 corpus, not the selected development/held-out material. Changing the fixture query disables its relevance grades.")
    question=st.text_area("Diagnostic research question",value="retrieval therapy",max_chars=1000,disabled=running or category!="Diagnostics")
    with st.expander("Reviewed corpus, public questions and evaluation inputs",expanded=category!="Diagnostics"):
        corpus=st.selectbox("Corpus snapshot",["Not selected",*corpora],disabled=running or category=="Diagnostics")
        questions=pick_file("Research question set",files,key="comparison-questions",disabled=running or category=="Diagnostics")
        factual=pick_file("Public factual probes (no answers)",files,key="comparison-facts",disabled=running or category=="Diagnostics")
        labels=pick_file("Independent relevance labels",files,key="comparison-labels",disabled=running or category=="Diagnostics")
        review=st.selectbox("Reviewed annotation bundle",["Not selected","data/prepilot-development-v1"],disabled=running or category=="Diagnostics")
        approval=pick_file("Pinned coordinator approval",files,key="comparison-approval",disabled=running or category=="Diagnostics")
    a,b,c=st.columns(3)
    actions=int(a.number_input("Actions per method",1,100,3,disabled=running or kind=="Long-horizon pressure"));trials=int(b.number_input("Repeated trials",1,20,1,disabled=running))
    seed_text=c.text_input("Trial seeds (comma separated)",value="0",disabled=running)
    with st.expander("Shared budgets and output",expanded=True):
        calls=int(st.number_input("Total model calls per method",2,100,5,disabled=running or kind=="Long-horizon pressure"))
        inspections=int(st.number_input("Inspection ceiling",1,10000,30,disabled=running or kind=="Long-horizon pressure"))
        page_size=int(st.number_input("Papers per acquisition",1,100,5,disabled=running or kind=="Long-horizon pressure"))
        active=int(st.number_input("Active working-context study tokens",512,16000,4096,disabled=running))
        full=int(st.number_input("Full input study-token ceiling",1024,64000,8192,disabled=running))
        tokens=int(st.number_input("Per-method cumulative token ceiling",1000,10000000,10000000 if kind=="Long-horizon pressure" else 70000,disabled=running,key="tokens-pressure" if kind=="Long-horizon pressure" else "tokens-comparison"))
        dollars=float(st.number_input("Per-method dollar ceiling (0 unset)",0.0,1000.0,0.0,disabled=running or planner!="hosted"))
        global_tokens=int(st.number_input("Global token ceiling",1000,100000000,10000000,disabled=running))
        global_dollars=float(st.number_input("Global dollar ceiling (0 unset)",0.0,10000.0,0.0,disabled=running or planner!="hosted"))
        output=st.text_input("New experiment name",value="comparison",disabled=running)
    pressure="low";observations=None;position=0;distractors=350;correction=None;archive=True
    if kind=="Long-horizon pressure":
        st.caption("Scripted diagnostic only. Pressure is validated from measured distinct-source and incremental tokens, not paper count.")
        st.caption("The stream uses one paper per observation, horizon+1 actions/rounds and horizon×4+16 calls per arm. Benchmark action/call controls are disabled; token ceilings above apply.")
        pressure=st.selectbox("Context pressure",["low","medium","high"],disabled=running)
        observations=int(st.number_input("Observation sequence length",2,1000,{"low":4,"medium":32,"high":256}[pressure],key="observations-"+pressure,disabled=running))
        position=int(st.number_input("Important evidence position (zero based)",0,observations-1,0,disabled=running))
        distractors=int(st.slider("Distracting source characters per observation",80,350,350,disabled=running))
        archive=st.checkbox("Shared external archive available",value=True,disabled=running)
        if st.checkbox("Introduce corrected / contradictory later evidence",disabled=running):
            if position<observations-1:correction=int(st.number_input("Correction observation index",position+1,observations-1,observations-1,disabled=running))
            else:st.warning("Move important evidence earlier to allow a later correction.")
    paid=st.checkbox("I confirm this hosted run may consume paid API credits",value=False,disabled=running or planner!="hosted")
    if category=="Diagnostics":
        corpus="Not selected";questions=factual=labels=approval=None;review="Not selected"
    try:
        seeds=tuple(int(s.strip()) for s in seed_text.split(",") if s.strip())
        config=DashboardConfig(kind=kind,category=category,methods=tuple(selected),planner=planner,model=model or None,
            corpus=None if corpus=="Not selected" else corpus,questions=questions,factual_questions=factual,labels=labels,
            review_bundle=None if review=="Not selected" else review,approval=approval,question=question,output_name=output,
            trials=trials,seeds=seeds,action_budgets=(actions,),budget=Budget(max_actions=actions,max_rounds=actions,max_inspected=inspections,
                max_llm_calls=calls,page_size=page_size,output_k=10,max_context_chars=64000),
            limits=InferenceLimits(tokens,(dollars or None) if planner=="hosted" else None,active,full),global_tokens=global_tokens,global_dollars=(global_dollars or None) if planner=="hosted" else None,
            paid_confirmed=paid,pressure=pressure,active_tokens=active,observations=observations,evidence_position=position,
            distractor_chars=distractors,correction_at=correction,archive_reads=archive)
        config.validate();gate=readiness(config,root,settings)
    except (ValueError,TypeError):
        st.error("Check method selection, seed count, token ceilings and numerical settings.");return
    with st.expander("Configuration equivalence and maximum usage",expanded=True):
        qcount=1
        if config.questions and category!="Diagnostics":
            try:qcount=len(json.loads(allowed_path(config.questions,root).read_text()))
            except (ValueError,OSError,TypeError):pass
        preview=equivalence(config,settings,questions=qcount)
        st.dataframe(preview.pop("methods"),hide_index=True)
        for warning in preview.pop("warnings"):st.warning(warning)
        st.json(redact(preview,secrets),expanded=False)
        st.caption("Price estimates require configured positive prices. Token estimates and observed limits are not prepaid guarantees; frozen-corpus comparisons make zero live OpenAlex calls.")
        st.json({"spec_sha256":approval_digest(config),"provider_profile_sha256":profile_digest(settings,config.model),"source_sha256":source_digest()},expanded=False)
    if not gate["ready"]:
        st.warning("NOT READY FOR THIS EMPIRICAL RUN · "+"; ".join(gate["reasons"]))
        st.caption("Complete independent review and coordinator approval. A paid-confirmation checkbox cannot bypass scientific readiness.")
    if job:st.caption("This submission is already recorded. Choose Prepare another experiment for an explicit new run.")
    if st.button("Run Comparison",type="primary",disabled=bool(job) or not gate["ready"] or planner=="hosted" and not paid):
        try:
            token=st.session_state.setdefault("dashboard_submission",str(uuid.uuid4()))
            st.session_state.dashboard_job=start_dashboard(config,settings,root=root,submission_id=token)
        except (ValueError,OSError):st.error("Cannot start this configuration. Check approved paths, readiness and budgets.")
        else:st.rerun()
    if job:
        st.fragment(run_every=1.0 if running else None)(render_progress)(job,root,secrets)
