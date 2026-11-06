"""Safe experiment adapter and read-only artifact views; no research ranker here."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

import pandas as pd

from .config import Settings
from .context_planner import InferenceLimits
from .controller import Budget
from .controlled import ARCHITECTURES
from .corpus import LocalCorpus, atomic_json
from .development_review import validate_review
from .execution_control import ExperimentStopped
from .experiment_spec import BenchmarkSpec
from .git_provenance import git_provenance
from .experiments import run_experiments
from .llm import DeterministicPlanner, HostedJSONPlanner
from .research import ObservedDatabase, redact
from .storage import Database

LABELS={"standard":"Standard OpenAlex", "llm":"Conventional LLM", "llm_graph":"Graph-enabled LLM",
        "clm":"CLM-inspired LLM", "clm_graph":"Graph + CLM LLM"}
CATEGORIES=("Diagnostics","Development pilot","Final research evaluation")
KINDS=("Comparison","Graph ablation","CLM ablation","Factual evaluation","Five-method comparison","Long-horizon pressure","G1–G5 / C1–C7 diagnostics")
QUALITY={"Precision@10":"precision_at_10","nDCG@10":"ndcg_at_10","Factual answer accuracy":"task_overall_answer_accuracy",
    "Answer coverage":"task_answer_coverage","Grounded completion":"task_overall_task_completion",
    "Supported claims":"supported_claim_rate","Unsupported claims":"unsupported_claim_rate","Contradictions":"contradicted_claim_rate",
    "Unique relevant papers":"unique_relevant_papers","Relevant papers per action":"useful_papers_per_action"}
OPERATIONAL={"Retrieval actions":"actions","Total tokens (reported)":"total_reported_tokens","Estimated tokens (separate)":"estimated_unreported_tokens",
    "Context-edit overhead (s)":"context_management_seconds","API cost (USD)":"total_api_cost_usd","Runtime (s)":"end_to_end_seconds",
    "Provider failures":"llm_failures","Fallbacks":"fallbacks"}


def allowed_path(value, root, *, exists=True):
    root=Path(root).resolve()
    original=Path(value)
    path=(original if original.is_absolute() else root/original).resolve()
    if not any(path.is_relative_to(root/p) for p in ("data","benchmarks","docs/prepilot")):
        raise ValueError("Choose an artifact under approved project data/benchmark locations")
    if exists and not path.exists():raise ValueError("Selected artifact does not exist")
    return path


def output_directory(name,root):
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",name):
        raise ValueError("Output name must contain only letters, digits, underscores or hyphens")
    root=Path(root).resolve();base=root/"data/experiment-dashboard"
    if base.resolve()!=base:raise ValueError("Output location cannot be redirected")
    return base/(name+"-"+str(uuid.uuid4()))


@dataclass(frozen=True)
class DashboardConfig:
    kind: str = "Comparison"
    category: str = "Diagnostics"
    methods: tuple[str,...] = ("standard","llm","llm_graph","clm","clm_graph")
    planner: str = "deterministic"
    corpus: str | None = None
    questions: str | None = None
    question: str = "retrieval therapy"
    factual_questions: str | None = None
    labels: str | None = None
    claim_judgments: str | None = None
    review_bundle: str | None = None
    approval: str | None = None
    output_name: str = "comparison"
    trials: int = 1
    seeds: tuple[int,...] = (0,)
    action_budgets: tuple[int,...] = (3,)
    budget: Budget = field(default_factory=lambda:Budget(max_actions=3,max_rounds=3,max_inspected=30,page_size=5,output_k=10,max_llm_calls=5))
    limits: InferenceLimits = field(default_factory=lambda:InferenceLimits(max_total_tokens=70000,max_active_tokens=4096,max_input_tokens=8192))
    global_tokens: int = 1000000
    global_dollars: float | None = None
    paid_confirmed: bool = False
    model: str | None = None
    pressure: str = "low"
    active_tokens: int = 4096
    observations: int | None = None
    evidence_position: int = 0
    distractor_chars: int = 350
    correction_at: int | None = None
    archive_reads: bool = True

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if isinstance(data.get("budget"),dict):data["budget"]=Budget(**data["budget"])
        if isinstance(data.get("limits"),dict):data["limits"]=InferenceLimits(**data["limits"])
        for name in ("methods","seeds","action_budgets"):
            if name in data:data[name]=tuple(data[name])
        return cls(**data)

    def validate(self):
        if self.kind not in KINDS or self.category not in CATEGORIES or self.planner not in ("deterministic","hosted"):
            raise ValueError("Unknown experiment kind/category/planner")
        if not self.methods or len(set(self.methods))!=len(self.methods) or any(m not in ARCHITECTURES for m in self.methods):
            raise ValueError("Select distinct supported architectures")
        if type(self.trials) is not int or not 1<=self.trials<=20 or len(self.seeds)!=self.trials or any(type(s) is not int or s<0 for s in self.seeds):
            raise ValueError("Provide one nonnegative seed for each trial (maximum 20)")
        if not self.action_budgets or any(type(a) is not int or not 1<=a<=100 for a in self.action_budgets):
            raise ValueError("Invalid retrieval action budgets")
        if type(self.global_tokens) is not int or self.global_tokens<1 or self.global_dollars is not None and (
                isinstance(self.global_dollars,bool) or not math.isfinite(self.global_dollars) or self.global_dollars<=0):
            raise ValueError("Invalid global resource ceiling")
        if not isinstance(self.question,str) or not 1<=len(self.question.strip())<=1000:
            raise ValueError("Enter a research question")
        if self.kind=="Five-method comparison" and set(self.methods)!=set(ARCHITECTURES):
            raise ValueError("Five-method comparison requires all five architectures")
        if self.kind=="Long-horizon pressure" and ("standard" in self.methods or self.pressure not in ("low","medium","high")):
            raise ValueError("Pressure requires agent architectures and a valid pressure level")
        if self.kind in ("Long-horizon pressure","G1–G5 / C1–C7 diagnostics") and (self.category!="Diagnostics" or self.planner!="deterministic"):
            raise ValueError("These correctness fixtures are offline diagnostics only")


def profile_digest(settings,model=None):
    return hashlib.sha256(json.dumps(dict(model=model or settings.llm_model,temperature=settings.llm_temperature,
        reasoning_effort=settings.llm_reasoning_effort,max_output_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,input_price=settings.llm_input_price,output_price=settings.llm_output_price,
        endpoint_hash=hashlib.sha256(settings.llm_base_url.encode()).hexdigest()),sort_keys=True).encode()).hexdigest()


def source_digest():
    return hashlib.sha256(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")},sort_keys=True).encode()).hexdigest()


def capture_review_snapshot(config,root,directory,settings):
    """Keep the actual reviewed inputs, not just a boolean readiness assertion."""
    if config.category=="Diagnostics":return
    import shutil
    bundle=allowed_path(config.review_bundle,root)
    manifest=json.loads((bundle/"review-manifest.json").read_text())
    names=set(manifest["initial_immutable_sha256"])|{"review-manifest.json","relevance-worksheet.csv","evidence-adequacy-worksheet.csv"}
    destination=directory/"review-snapshot"
    for name in names:
        source=allowed_path(bundle/name,root)
        if not source.is_relative_to(bundle):raise ValueError("Review input escapes its bundle")
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    review=validate_review(destination)
    if review["independent_judgments"]!="PASS" or review["evidence_adequacy"]!="PASS":raise ValueError("Review changed before snapshot commitment")
    approval=json.loads(allowed_path(config.approval,root).read_text())
    atomic_json(directory/"approval-snapshot.json",{k:approval[k] for k in (
        "approved","approver","approved_at","category","questions_sha256","corpus_sha256","spec_sha256","source_sha256","provider_profile_sha256")})
    snapshot=json.loads((directory/"approval-snapshot.json").read_text())
    snapshot["heldout_independence_attested"]=bool(approval.get("heldout_independence_attested"))
    atomic_json(directory/"approval-snapshot.json",snapshot)


def saved_approval_valid(directory,dashboard,manifest):
    try:
        if (directory/"approval-snapshot.json").is_symlink():return False
        snapshot=directory/"review-snapshot"
        if any(p.is_symlink() for p in snapshot.rglob("*")):return False
        review_manifest=json.loads((snapshot/"review-manifest.json").read_text())
        if any(not (snapshot/name).resolve().is_relative_to(snapshot.resolve()) for name in review_manifest["initial_immutable_sha256"]):return False
        review=validate_review(directory/"review-snapshot")
        approval=json.loads((directory/"approval-snapshot.json").read_text())
        reviewed=json.loads((directory/"review-snapshot/review-manifest.json").read_text())
        if review["independent_judgments"]!="PASS" or review["evidence_adequacy"]!="PASS":return False
        if not approval["approved"] or not approval.get("approver") or not approval.get("approved_at") or approval["category"]!=dashboard["category"]:return False
        if approval["spec_sha256"]!=approval_digest(DashboardConfig.from_dict(dashboard["configuration"])):return False
        if dashboard["category"]=="Final research evaluation" and (not 10<=len(manifest["questions"])<=20 or manifest.get("trials",0)<3
                or any(q.get("split")!="heldout" for q in manifest["questions"]) or not approval.get("heldout_independence_attested")):return False
        if approval["provider_profile_sha256"]!=dashboard.get("provider_profile_sha256"):return False
        hashes=manifest["source_sha256"]
        if approval["source_sha256"]!=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest():return False
        source=(directory/"benchmark/source") if (directory/"benchmark/source").is_dir() else directory/"source"
        for name,digest in hashes.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+\.py",name):return False
            file=source/name
            if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:return False
        if not review_labels_match(directory/"review-snapshot",manifest.get("judgments",{})):return False
        return (reviewed["questions"]==manifest["questions"] and approval["questions_sha256"]==manifest["questions_sha256"]
                and approval["corpus_sha256"]==manifest["corpus_manifest"]["sha256"])
    except (OSError,ValueError,KeyError,TypeError):return False


def review_labels_match(bundle,data):
    frame=pd.read_csv(Path(bundle)/"relevance-worksheet.csv",dtype=str).fillna("")
    reviewed={(r.query_id,r.paper_id):int(r.grade) for r in frame.itertuples()}
    imported={(r["query_id"],r["paper_id"]):r["grade"] for r in data.get("judgments",[])}
    return data.get("source")=="human_import" and imported==reviewed


def claim_review_valid(data,claims_sha256):
    reviewers=data.get("reviewer_ids")
    return (isinstance(claims_sha256,str) and re.fullmatch(r"[0-9a-f]{64}",claims_sha256) is not None
        and data.get("claims_sha256")==claims_sha256 and data.get("review_status")=="verified"
        and data.get("independent_human_review") is True and data.get("adjudicated") is True
        and isinstance(data.get("approver"),str) and bool(data["approver"].strip())
        and isinstance(reviewers,list) and bool(reviewers)
        and all(isinstance(r,str) and bool(r.strip()) for r in reviewers))


def readiness(config,root,settings=None):
    """Fail closed; a confirmation checkbox never substitutes for human review."""
    if config.category=="Diagnostics":
        return {"ready":config.planner=="deterministic","reasons":[] if config.planner=="deterministic" else ["Diagnostics cannot make hosted calls"]}
    reasons=[]
    try:
        bundle=allowed_path(config.review_bundle or "",root)
        review=validate_review(bundle)
        if review["independent_judgments"]!="PASS" or review["evidence_adequacy"]!="PASS":reasons.append("Independent human review or factual verification is incomplete")
        manifest=json.loads((bundle/"review-manifest.json").read_text())
        questions=allowed_path(config.questions or "",root)
        actual=json.loads(questions.read_text())
        if actual!=manifest["questions"]:reasons.append("Question set does not match the reviewed bundle")
        corpus=LocalCorpus(allowed_path(config.corpus or "",root))
        if corpus.fingerprint!=manifest["corpus_manifest"]["sha256"]:reasons.append("Corpus differs from reviewed snapshot")
        if not config.factual_questions or json.loads(allowed_path(config.factual_questions,root).read_text())!=manifest["factual_questions"]:
            reasons.append("Matching reviewed public factual probes are required")
        approval_path=allowed_path(config.approval or "",root)
        approval=json.loads(approval_path.read_text())
        if not approval.get("approved") or not approval.get("approver") or not approval.get("approved_at") or approval.get("category")!=config.category:
            reasons.append("Coordinator protocol approval is missing")
        pins={"questions_sha256":hashlib.sha256(questions.read_bytes()).hexdigest(),"corpus_sha256":corpus.fingerprint,
            "spec_sha256":approval_digest(config),"source_sha256":source_digest()}
        if settings is not None:pins["provider_profile_sha256"]=profile_digest(settings,config.model)
        if any(approval.get(k)!=v for k,v in pins.items()):reasons.append("Approval does not pin this exact experiment configuration")
        # Importable labels must equal the actual adjudicated review rows.
        label_data=json.loads(allowed_path(config.labels or "",root).read_text())
        frame=pd.read_csv(bundle/"relevance-worksheet.csv",dtype=str).fillna("")
        reviewed={(r.query_id,r.paper_id):int(r.grade) for r in frame.itertuples()}
        imported={(r["query_id"],r["paper_id"]):r["grade"] for r in label_data["judgments"]}
        if imported!=reviewed or label_data.get("source")!="human_import":reasons.append("Relevance labels do not match adjudicated human worksheets")
        if config.category=="Final research evaluation" and (not 10<=len(actual)<=20 or config.trials<3 or
                any(q.get("split")!="heldout" for q in actual) or not approval.get("heldout_independence_attested")):
            reasons.append("Final evaluation needs 10–20 approved held-out questions, three repeats and holdout attestation")
    except (OSError,ValueError,KeyError,TypeError):
        reasons.append("A complete reviewed bundle, imported labels and pinned coordinator approval are required")
    return {"ready":not reasons,"reasons":list(dict.fromkeys(reasons))}


def approval_digest(config):
    data=asdict(config)
    for key in ("paid_confirmed","approval","output_name"):data.pop(key,None)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def reserved_question(question,root):
    path=Path(root)/"data/study/reserved-questions.json"
    try:
        rows=json.loads(path.read_text())
        return " ".join(question.casefold().split()) in {" ".join(q["question"].casefold().split()) for q in rows}
    except (OSError,ValueError,KeyError,TypeError):return False


def equivalence(config, settings, *, questions=1):
    warnings=[];agents=set(config.methods)-{"standard"}
    if agents and agents!={"llm","llm_graph","clm","clm_graph"}:warnings.append("Selected subset supports conditional contrasts; full graph/CLM main effects and interaction require all four agent cells")
    if "standard" in config.methods:warnings.append("Standard uses its inspection allowance in one search and emits no model claims")
    warnings.extend(["Graph/CLM context content and management-call use differ by design; final ranking is shared lexical cosine",
        "Trial seeds are recorded; hosted sampling is not claimed seeded or exactly reproducible"])
    calls=questions*config.trials*len(agents)*len(config.action_budgets)*config.budget.max_llm_calls
    if config.kind=="Long-horizon pressure":
        from .context_pressure import CONDITIONS
        calls=config.trials*len(agents)*((config.observations or CONDITIONS[config.pressure])*4+16)
    if config.planner=="deterministic":
        price=0.0
    else:price=None
    upper_tokens=calls*((config.limits.max_input_tokens or config.budget.max_context_chars)+settings.llm_max_tokens)
    if config.planner=="hosted" and settings.llm_input_price>0 and settings.llm_output_price>0:
        price=calls*((config.limits.max_input_tokens or config.budget.max_context_chars)*settings.llm_input_price+settings.llm_max_tokens*settings.llm_output_price)/1e6
    return dict(methods=[dict(method=m,graph=m.endswith("_graph"),context="none" if m=="standard" else "clm" if m.startswith("clm") else "conventional") for m in config.methods],
        model=config.model or settings.llm_model if config.planner=="hosted" else "scripted engineering model",model_settings=dict(temperature=settings.llm_temperature,reasoning_effort=settings.llm_reasoning_effort,max_output_tokens=settings.llm_max_tokens) if config.planner=="hosted" else {"protocol":"SCRIPTED ENGINEERING ONLY","hosted_settings_not_applied":True},
        budget=asdict(config.budget) if config.kind!="Long-horizon pressure" else dict(observations=config.observations or CONDITIONS[config.pressure],
            actions_and_rounds=(config.observations or CONDITIONS[config.pressure])+1,per_arm_calls=(config.observations or CONDITIONS[config.pressure])*4+16,page_size=1),
        limits=asdict(config.limits),max_model_calls=calls,estimated_max_tokens=upper_tokens,
        estimated_max_cost_usd=price,max_hosted_calls=calls if config.planner=="hosted" else 0,
        live_openalex_attempts=0,global_tokens=config.global_tokens,global_dollars=config.global_dollars,warnings=warnings)


class DashboardJob:
    def __init__(self,directory,config):
        self.directory=Path(directory);self.config=config;self.cancelled=threading.Event();self.lock=threading.RLock();self.thread=None
        self.state=dict(status="starting",directory=str(directory),recorded_events=0,last_event=None,error=None)

    def update(self,**values):
        with self.lock:
            self.state.update(values);atomic_json(self.directory/"status.json",self.state)

    def snapshot(self):
        with self.lock:state=deepcopy(self.state)
        path=self.directory/"benchmark/metrics.csv"
        try:state["completed_cells"]=len(pd.read_csv(path))
        except (OSError,ValueError,pd.errors.EmptyDataError):state["completed_cells"]=0
        state["recorded_inference_snapshots"]=sum(1 for _ in self.directory.rglob("request-*.json"))
        return state

    def observe(self,event):
        with self.lock:
            self.state["recorded_events"]+=1;self.state["last_event"]=event.get("kind")
            with (self.directory/"activity.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")

    def cancel(self):
        with self.lock:
            if self.state.get("finished") or self.state["status"] not in ("starting","running","cancellation_requested"):return False
            self.cancelled.set();self.update(status="cancellation_requested")
            return True

    def join(self,timeout=None):
        if self.thread:self.thread.join(timeout)


_JOBS={};_JOBS_LOCK=threading.Lock()


def start_dashboard(config,settings,*,root,submission_id,planner_transport=None):
    config.validate();root=Path(root).resolve()
    with _JOBS_LOCK:
        key=(str(root),submission_id)
        if key in _JOBS:return _JOBS[key]
        gate=readiness(config,root,settings)
        if not gate["ready"]:raise ValueError("Scientific readiness blocked: "+"; ".join(gate["reasons"]))
        if config.planner=="hosted" and not config.paid_confirmed:raise ValueError("Explicit paid-run confirmation is required")
        if config.planner=="hosted" and not all((settings.llm_api_key,settings.llm_base_url,config.model or settings.llm_model)):
            raise ValueError("Hosted planner configuration is incomplete")
        if config.planner=="hosted" and config.global_dollars is not None and (settings.llm_input_price<=0 or settings.llm_output_price<=0):raise ValueError("A global dollar ceiling requires known positive prices")
        secrets=tuple(v for v in (settings.api_key,settings.llm_api_key,settings.email) if v)
        if any(s in json.dumps(asdict(config)) for s in secrets):raise ValueError("Remove credentials from experiment inputs")
        for name in ("corpus","questions","factual_questions","labels","claim_judgments","review_bundle","approval"):
            if getattr(config,name):allowed_path(getattr(config,name),root)
        directory=output_directory(config.output_name,root);directory.mkdir(parents=True,exist_ok=False)
        capture_review_snapshot(config,root,directory,settings)
        job=DashboardJob(directory,config)
        atomic_json(directory/"dashboard.json",dict(schema_version=1,git=git_provenance(root),configuration=asdict(config),category=config.category,
            readiness=gate,hosted_sampling_seed_supported=False,configuration_digest=approval_digest(config),
            provider_profile_sha256=profile_digest(settings,config.model)))
        _JOBS[key]=job
        job.thread=threading.Thread(target=_execute,args=(job,replace(settings,llm_model=config.model or settings.llm_model),root,secrets,planner_transport),daemon=True)
        job.thread.start();return job


def _execute(job,settings,root,secrets,transport):
    from .context_pressure import run_pressure
    from .controlled_diagnostics import fixture,run_graph_case
    from .corpus import freeze
    from .research_assessment import score_research
    config=job.config
    totals=dict(tokens=0,cost=0.0,calls=0)
    class GuardedPlanner(HostedJSONPlanner):
        def _call(self,system,payload):
            from .token_budget import input_tokens
            if job.cancelled.is_set():raise ExperimentStopped()
            predicted=input_tokens(system,payload)
            reserve=settings.llm_max_tokens
            projected=(predicted*settings.llm_input_price+reserve*settings.llm_output_price)/1e6
            if totals["tokens"]+predicted+reserve>config.global_tokens or config.global_dollars is not None and totals["cost"]+projected>config.global_dollars:
                raise ExperimentStopped("global_budget_exhausted")
            before=asdict(self.usage)
            try:return super()._call(system,payload)
            finally:
                after=asdict(self.usage)
                known=after["usage_unreported_calls"]==before["usage_unreported_calls"]
                incoming=after["input_tokens"]-before["input_tokens"] if known else after["estimated_input_tokens"]-before["estimated_input_tokens"]
                outgoing=after["output_tokens"]-before["output_tokens"] if known else after["estimated_output_tokens"]-before["estimated_output_tokens"]
                totals["tokens"]+=incoming+outgoing;totals["calls"]+=after["calls"]-before["calls"]
                totals["cost"]+=(incoming*settings.llm_input_price+outgoing*settings.llm_output_price)/1e6
                known_price=bool(settings.llm_input_price and settings.llm_output_price)
                job.update(global_usage=dict(tokens=totals["tokens"],calls=totals["calls"],cost_usd=totals["cost"] if known_price else None,cost_known=known_price))
    def cancelled():
        if job.cancelled.is_set():return True
        if totals["tokens"]>=config.global_tokens:raise ExperimentStopped("global_budget_exhausted")
        return False
    def observe(event):
        job.observe(redact(event,secrets))
        if event.get("kind")=="context_model_call" and config.planner=="deterministic":
            usage=event.get("usage_delta",{})
            totals["tokens"]+=usage.get("estimated_input_tokens",0)+usage.get("estimated_output_tokens",0)
            totals["calls"]+=1
            job.update(global_usage=dict(tokens=totals["tokens"],calls=totals["calls"],cost_usd=0.0,cost_known=True,usage_basis="scripted estimates"))
        if event.get("kind")=="context_model_call":
            if job.cancelled.is_set():raise ExperimentStopped()
            if totals["tokens"]>=config.global_tokens:raise ExperimentStopped("global_budget_exhausted")
    try:
        job.update(status="running")
        if config.kind=="G1–G5 / C1–C7 diagnostics" or config.kind=="Factual evaluation" and config.category=="Diagnostics":
            graph=[]
            for case in (("G1","G2","G3","G5") if config.kind!="Factual evaluation" else ()):
                if cancelled():raise ExperimentStopped()
                graph.append(run_graph_case(case,job.directory/case))
                atomic_json(job.directory/"graph-diagnostics.json",graph)
            # Trusted static test targets, no user-controlled command strings.
            targets=["tests/test_prepilot_protocol.py","tests/test_research_assessment.py"] if config.kind=="Factual evaluation" else ["tests/test_clm_context.py","tests/test_controlled_graph.py"]
            command=[sys.executable,"-m","pytest",*targets,
                     "--override-ini","addopts=","-q","--basetemp",str(job.directory/"pytest-data")]
            with (job.directory/"diagnostics.log").open("w") as log:
                import os,signal
                process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=="posix")
                while process.poll() is None:
                    if job.cancelled.wait(.2):
                        if os.name=="posix":os.killpg(process.pid,signal.SIGTERM)
                        else:process.terminate()
                        process.wait();raise ExperimentStopped()
            atomic_json(job.directory/"diagnostics.json",dict(kind="OFFLINE CORRECTNESS TESTS",command=command,exit_code=process.returncode,
                note="G4 tested with graph fixtures; C1–C7 are the existing test targets"))
            if process.returncode:raise RuntimeError("Diagnostic checks failed")
        elif config.kind=="Long-horizon pressure":
            for trial,seed in enumerate(config.seeds):
                if cancelled():raise ExperimentStopped()
                result=run_pressure(config.pressure,job.directory/f"pressure-trial-{trial}",seed=seed,active_tokens=config.active_tokens,
                    observations=config.observations,evidence_position=config.evidence_position,distractor_chars=config.distractor_chars,
                    correction_at=config.correction_at,archive_reads=config.archive_reads,architectures=config.methods,
                    return_invalid=True,cancel_check=cancelled,global_token_limit=max(1,config.global_tokens-totals["tokens"]),
                    full_input_tokens=config.limits.max_input_tokens,per_arm_tokens=config.limits.max_total_tokens)
                totals["tokens"]+=result["total_study_tokens_used"]
                job.update(completed_trials=trial+1,last_pressure_valid=result["pressure_valid"])
        else:
            corpus=config.corpus;questions=config.questions
            labels=config.labels
            if config.category=="Diagnostics":
                question,papers,_,positives,_=fixture("G1")
                corpus=str(job.directory/"synthetic-corpus");freeze(papers,corpus,acquisition=[{"source":"SYNTHETIC DASHBOARD DIAGNOSTIC ONLY"}])
                questions=str(job.directory/"questions.json");atomic_json(Path(questions),[dict(id="diagnostic-q",question=config.question)])
                labels=str(job.directory/"synthetic-labels.json")
                atomic_json(Path(labels),dict(source="curated",scope="exhaustive",fixture_only=True,judgments=[
                    dict(query_id="diagnostic-q",paper_id=p.id,grade=positives.get(int(p.id.rsplit('W',1)[-1]),0)) for p in papers]))
                if config.question.strip()!=question:
                    labels=None  # fixture grades do not label arbitrary new questions
            if not corpus or not questions:raise ValueError("Select a corpus and question set")
            spec=BenchmarkSpec(str(allowed_path(corpus,root)),str(allowed_path(questions,root)),str(job.directory/"benchmark"),
                methods=config.methods,action_budgets=config.action_budgets,trials=config.trials,trial_seeds=config.seeds,
                budget=config.budget,inference_limits=config.limits,labels=str(allowed_path(labels,root)) if labels else None,
                factual_questions=str(allowed_path(config.factual_questions,root)) if config.factual_questions else None,
                allow_deterministic=config.planner=="deterministic")
            atomic_json(job.directory/"resolved-spec.json",spec.to_dict())
            factory=(lambda:GuardedPlanner(settings,transport=transport)) if config.planner=="hosted" else DeterministicPlanner
            with ObservedDatabase(str(job.directory/"metadata.sqlite"),observe) as db:
                run_experiments(spec.corpus,spec.questions,db,factory,**spec.backend_kwargs(),cancel_check=cancelled)
            if config.claim_judgments:
                score_research(spec.output,labels_path=spec.labels,claim_judgments_path=str(allowed_path(config.claim_judgments,root)))
        job.update(status="completed")
    except ExperimentStopped as exc:job.update(status=exc.reason)
    except Exception as exc:job.update(status="failed",error={"type":type(exc).__name__,"message":"Execution failed; raw and incomplete artifacts were preserved."})
    finally:
        job.update(finished=True)


def saved_experiments(root):
    root=Path(root).resolve();records={}
    for filename in ("dashboard.json","experiment.json","pressure.json","diagnostics.json"):
        for p in (root/"data").rglob(filename):
            try:
                allowed_path(p,root)
                parent=p.parent
                # Dashboard roots encapsulate their nested raw artifacts.
                dashboard=next((a for a in (parent,*parent.parents) if a.is_relative_to(root/"data") and (a/"dashboard.json").is_file()),None)
                chosen=dashboard or parent
                records[str(chosen.relative_to(root))]=chosen.stat().st_mtime
            except (OSError,ValueError):continue
    return sorted(records,key=records.get,reverse=True)


def _json(path,default=None):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):return default


def safe_json(path,root,default=None):
    try:return _json(allowed_path(path,root),default)
    except ValueError:return default


def read_comparison(directory,root):
    """Read only. No scoring, plotting command, or model/retrieval call occurs."""
    directory=allowed_path(directory,root)
    dashboard=safe_json(directory/"dashboard.json",root,{})
    benchmark=directory/"benchmark" if (directory/"benchmark/experiment.json").is_file() else directory
    manifest=safe_json(benchmark/"experiment.json",root,{})
    category=dashboard.get("category") or ("Diagnostics" if manifest.get("deterministic_smoke") else "UNVERIFIED SAVED EXPERIMENT")
    rows=[]
    try:raw=pd.read_csv(allowed_path(benchmark/"metrics.csv",root),dtype={"query_id":str}).to_dict("records")
    except (OSError,ValueError,pd.errors.EmptyDataError):raw=[]
    research={}
    try:
        for r in pd.read_csv(allowed_path(benchmark/"research-metrics.csv",root),dtype={"query_id":str}).to_dict("records"):
            research[tuple(r[k] for k in ("query_id","method","variant","action_budget","trial"))]=r
    except (OSError,ValueError,KeyError,pd.errors.EmptyDataError):pass
    relevance=bool(manifest.get("judgments",{})) and manifest["judgments"].get("source") in ("curated","human_import")
    provenance=safe_json(benchmark/"assessment-provenance.json",root,{})
    factual=bool(provenance.get("claim_judgments_sha256"))
    acquisitions=manifest.get("corpus_manifest",{}).get("acquisition",[])
    synthetic=bool(acquisitions) and all("SYNTHETIC" in str(a.get("source","")).upper() for a in acquisitions)
    fixture=category=="Diagnostics" and bool(manifest.get("deterministic_smoke")) and synthetic
    approved=category=="Diagnostics" or saved_approval_valid(directory,dashboard,manifest)
    if not approved:category="UNVERIFIED SAVED EXPERIMENT"
    relevance=relevance and approved and (category!="Diagnostics" or fixture and manifest.get("judgments",{}).get("fixture_only") is True)
    factual=factual and approved and (category!="Diagnostics" or fixture)
    if category!="Diagnostics":
        factual=factual and claim_review_valid(safe_json(directory/"claim-review-snapshot.json",root,{}),provenance.get("claim_judgments_sha256"))
    for r in raw:
        key=tuple(r.get(k) for k in ("query_id","method","variant","action_budget","trial"))
        merged={**r,**research.get(key,{})}
        for label,column in QUALITY.items():
            valid=relevance if column in ("precision_at_10","ndcg_at_10","unique_relevant_papers","useful_papers_per_action") else factual
            if factual and column=="task_overall_answer_accuracy" and merged.get("task_independent_judgment_coverage",0)<=0:valid=False
            if not valid:merged[column]=None
        merged["total_reported_tokens"]=r.get("input_tokens",0)+r.get("output_tokens",0)
        merged["estimated_unreported_tokens"]=r.get("estimated_input_tokens",0)+r.get("estimated_output_tokens",0)
        if "end_to_end_seconds" not in merged:merged["end_to_end_seconds"]=r.get("seconds")
        merged["evaluation_status"]=("SYNTHETIC DIAGNOSTIC LABELS" if relevance or factual else "DIAGNOSTIC · QUALITY NOT EVALUATED") if category=="Diagnostics" else "INDEPENDENTLY SCORED" if factual and relevance else "PARTIAL / NOT EVALUATED"
        merged["source_experiment"]=str(benchmark)
        identifier=str(r.get("run_id",""))
        local=benchmark/"runs"/identifier
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",identifier):raise ValueError("Unsafe run identifier")
            merged["artifact_directory"]=str(allowed_path(local if local.is_dir() else r["directory"],root))
        except (ValueError,KeyError):merged["artifact_directory"]=None
        usage=safe_json(Path(merged["artifact_directory"])/"result.json",root,{}).get("llm_usage",{}) if merged["artifact_directory"] else {}
        if usage.get("usage_unreported_calls",1):merged["total_reported_tokens"]=None
        rows.append(merged)
    pressure=[]
    paths=[directory/"pressure.json"] if (directory/"pressure.json").is_file() else list(directory.glob("pressure-trial-*/pressure.json"))
    for path in paths:
        report=safe_json(path,root,{})
        for row in report.get("rows",[]):
            pressure.append({**row,"pressure_valid":report.get("pressure_valid",False),"pressure_ratio":report.get("pressure_ratio"),
                "unique_source_pressure_ratio":report.get("unique_source_pressure_ratio"),"seed":report.get("seed"),"source_artifact":str(path)})
    completed_dirs={r["artifact_directory"] for r in rows}
    partial=[]
    for p in (benchmark/"runs").glob("*/state.json"):
        try:
            directory_path=str(allowed_path(p.parent,root))
            if directory_path not in completed_dirs:partial.append(directory_path)
        except ValueError:continue
    for p in directory.glob("pressure-trial-*/*/*/state.json"):
        try:
            directory_path=str(allowed_path(p.parent,root))
            if not any(r.get("directory")==directory_path for r in pressure):partial.append(directory_path)
        except ValueError:continue
    return dict(directory=str(directory),category=category,dashboard=dashboard,manifest=manifest,rows=rows,pressure=pressure,partial_runs=partial,
        status=safe_json(directory/"status.json",root,{}),factual_evaluated=factual,relevance_evaluated=relevance,assessment_provenance=provenance,
        diagnostics=safe_json(directory/"diagnostics.json",root),graph_diagnostics=safe_json(directory/"graph-diagnostics.json",root))


def context_records(directory,root):
    directory=allowed_path(directory,root);context=directory/"context"
    result=[]
    quotes=list(dict.fromkeys(c.get("evidence_quote") for c in safe_json(directory/"result.json",root,{}).get("evidence_records",[]) if c.get("evidence_quote")))
    for path in sorted(context.glob("request-*.json")):
        r=_json(allowed_path(path,root))
        if not r:continue
        payload=r.get("input",{});active=payload.get("active_context",json.dumps(payload.get("observation",{}),ensure_ascii=False))
        result.append(dict(call=r.get("call"),active=active,active_chars=len(active),
            active_tokens=r.get("study_tokens",{}).get("active_tokens"),full_input_tokens=r.get("study_tokens",{}).get("full_input_tokens"),
            recovery=payload.get("recovered_observation"),output=r.get("output"),system=r.get("system"),request=r))
        result[-1]["recorded_quotes_in_active"]=sum(q in active for q in quotes)
        result[-1]["recorded_quotes_in_recovery"]=sum(q in str(payload.get("recovered_observation",{})) for q in quotes)
    transitions=[_json(allowed_path(p,root)) for p in sorted(context.glob("transition-*.json"))]
    intents=[_json(allowed_path(p,root)) for p in sorted(context.glob("operation-*.json"))]
    return dict(requests=result,transitions=[r for r in transitions if r],intents=[r for r in intents if r])


def trace_records(directory,root):
    directory=allowed_path(directory,root)
    events=[]
    try:
        for line in allowed_path(directory/"events.jsonl",root).read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:continue  # incomplete last line remains on disk
    except (OSError,ValueError):pass
    signatures=[json.dumps([e.get("action"),e.get("arguments")],sort_keys=True) for e in events if e.get("kind")=="decision"]
    return dict(events=events,redundant_actions=len(signatures)-len(set(signatures)),result=safe_json(directory/"result.json",root,{}),
        state=safe_json(directory/"state.json",root,{}),graph_planner_inputs=[e.get("context") for e in events if e.get("kind")=="context" and "candidates" in e.get("context",{})])


def export_summary(loaded):
    rows=[]
    for r in loaded["rows"]:
        row={k:r.get(k) for k in ("query_id","method","variant","trial","seed","action_budget","model","corpus","evaluation_status","source_experiment","artifact_directory")}
        row.update(category=loaded["category"],source_manifest=str(Path(r["source_experiment"])/"experiment.json"),
            model_configuration=json.dumps(loaded["manifest"].get("provider_configuration",{}),sort_keys=True),
            budgets=json.dumps({"retrieval":loaded["manifest"].get("base_budget"),"inference":loaded["manifest"].get("inference_limits")},sort_keys=True))
        row.update({label:r.get(column) for label,column in {**QUALITY,**OPERATIONAL}.items()})
        for key,value in row.items():
            if value is None or isinstance(value,float) and not math.isfinite(value):row[key]="NOT EVALUATED"
            elif isinstance(value,str) and value.startswith(("=","+","-","@")):row[key]="'"+value
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def figure_export(loaded,metric):
    import os
    os.environ.setdefault("MPLCONFIGDIR","/tmp/openalex-dashboard-plots")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    frame=pd.DataFrame(loaded["rows"])
    if metric not in frame:raise ValueError("Metric is not available")
    frame[metric]=pd.to_numeric(frame[metric],errors="coerce");frame=frame.dropna(subset=[metric])
    if frame.empty:raise ValueError("Metric is NOT EVALUATED")
    means=frame.groupby(["method","variant","action_budget"])[metric].mean()
    fig=Figure(figsize=(9,5));ax=fig.subplots();ax.bar(["/".join(map(str,k)) for k in means.index],means.values)
    ax.tick_params(axis="x",rotation=30);ax.set_ylabel(metric)
    ax.set_title(loaded["category"]+" · descriptive trial means")
    fig.text(.01,.01,"Source: "+loaded["directory"]+" · evaluation status retained in exported CSV",fontsize=6)
    fig.tight_layout(rect=(0,.05,1,1));stream=io.BytesIO();fig.savefig(stream,format="png",dpi=150,
        metadata={"Description":json.dumps(dict(source=loaded["directory"],category=loaded["category"],metric=metric,
            evaluation_status=sorted({str(r.get("evaluation_status")) for r in loaded["rows"]}),configuration=loaded["manifest"]),default=str)})
    return stream.getvalue()


def assess_saved_copy(directory,root,*,labels=None,claims=None,claim_review=None):
    """Existing scorers write only into a fresh copy, never historical inputs."""
    import shutil
    from .experiments import score_experiment
    from .research_assessment import score_research
    loaded=read_comparison(directory,root)
    if loaded["category"]=="UNVERIFIED SAVED EXPERIMENT":
        raise ValueError("A recorded scientific approval is required for empirical scoring; diagnostic fixtures remain separate")
    if not labels and not claims:raise ValueError("Select independently supplied relevance or claim judgments")
    review_data=None
    if loaded["category"]!="Diagnostics":
        if labels and not review_labels_match(Path(loaded["directory"])/"review-snapshot",json.loads(allowed_path(labels,root).read_text())):
            raise ValueError("Labels differ from the independently adjudicated review snapshot")
        if claims:
            review_data=json.loads(allowed_path(claim_review or "",root).read_text())
            digest=hashlib.sha256(allowed_path(claims,root).read_bytes()).hexdigest()
            if not claim_review_valid(review_data,digest):raise ValueError("Independent claim review attestation is missing or mismatched")
            assessors=set(pd.read_csv(allowed_path(claims,root),dtype=str).fillna("").assessor)
            if not assessors or "" in assessors or not assessors<=set(review_data["reviewer_ids"]):raise ValueError("Claim reviewers do not match the attestation")
    source=allowed_path(loaded["rows"][0]["source_experiment"],root) if loaded["rows"] else None
    if source is None:raise ValueError("No benchmark rows are available to assess")
    for path in source.rglob("*"):
        if path.is_symlink():raise ValueError("Redirected artifacts cannot be copied for assessment")
    destination=output_directory("assessed-copy",root);destination.mkdir(parents=True)
    target=destination/"benchmark";shutil.copytree(source,target)
    frame=pd.read_csv(target/"metrics.csv",dtype={"query_id":str})
    for i,row in frame.iterrows():
        run=next(r for r in loaded["rows"] if r["run_id"]==row["run_id"])
        if not run["artifact_directory"]:raise ValueError("Raw run directory is unavailable")
        local=target/"runs"/str(row["run_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",str(row["run_id"])):raise ValueError("Unsafe run identifier")
        raw=allowed_path(run["artifact_directory"],root)
        if any(p.is_symlink() for p in raw.rglob("*")):raise ValueError("Redirected raw runs cannot be copied")
        if not local.exists():shutil.copytree(raw,local)
        frame.at[i,"directory"]=str(local)
    frame.to_csv(target/"metrics.csv",index=False)
    labels=str(allowed_path(labels,root)) if labels else None
    claims=str(allowed_path(claims,root)) if claims else None
    atomic_json(destination/"dashboard.json",{**loaded["dashboard"],"category":loaded["category"],"assessment_parent":str(source),
        "parent_manifest_sha256":hashlib.sha256((source/"experiment.json").read_bytes()).hexdigest()})
    parent=Path(loaded["directory"])
    if (parent/"review-snapshot").is_dir():shutil.copytree(parent/"review-snapshot",destination/"review-snapshot")
    if (parent/"approval-snapshot.json").is_file():shutil.copyfile(parent/"approval-snapshot.json",destination/"approval-snapshot.json")
    if (parent/"claim-review-snapshot.json").is_file():shutil.copyfile(parent/"claim-review-snapshot.json",destination/"claim-review-snapshot.json")
    if review_data is not None:atomic_json(destination/"claim-review-snapshot.json",review_data)
    if labels:score_experiment(str(target),labels)
    if claims:score_research(target,labels_path=labels,claim_judgments_path=claims)
    atomic_json(destination/"assessment-source.json",dict(parent=str(source),labels_sha256=hashlib.sha256(Path(labels).read_bytes()).hexdigest() if labels else None,
        claims_sha256=hashlib.sha256(Path(claims).read_bytes()).hexdigest() if claims else None))
    return destination


def main():
    """Equivalent guarded CLI execution of a dashboard specification."""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",required=True)
    parser.add_argument("--confirm-paid",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    data=json.loads(allowed_path(args.spec,root).read_text())
    config=DashboardConfig.from_dict(data.get("configuration",data))
    config=replace(config,paid_confirmed=args.confirm_paid)
    job=start_dashboard(config,Settings.from_env(),root=root,submission_id=str(uuid.uuid4()))
    try:
        while job.thread.is_alive():job.join(.5)
    except KeyboardInterrupt:
        job.cancel();job.join()
    state=job.snapshot();print(json.dumps(state))
    if state["status"]!="completed":raise SystemExit(1)


if __name__=="__main__":main()
