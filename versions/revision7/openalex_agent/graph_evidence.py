"""Graph-Visible V2: observed citation evidence and auditable fixed-capacity packing.

No corpus lookup, action selection, target keys or provider calls occur here.
"""
from collections import Counter
from copy import deepcopy
import hashlib,json
from .models import canonical_id,OPENALEX_SOURCES
from .ranking import relevance
from .selection import score_candidates
from .context_memory import serialized
from .token_budget import count

VERSION='graph-visible-v2'
MAX_GRAPH_TOKENS=768
MAX_EDGES=12


def identity(record):
    return hashlib.sha256(serialized(record).encode()).hexdigest()


def available_evidence(controller):
    """Use only currently retrieved sources; unresolved targets get no hidden metadata."""
    graph=controller.graph
    if graph is None:return [],0
    citations=graph.citation_graph()
    scores=relevance(controller.question,list(controller.papers.values()))
    anchors,_,_=controller._eligible()
    seeds=sorted(anchors,key=lambda p:(-scores.get(p,0),p))[:3]
    groups=[]
    for source in seeds:
        records=[]
        for target in sorted(citations.successors(source),key=lambda p:(-scores.get(p,0),p)):
            if source not in controller.papers or not graph._citation_verified(source,target):continue
            a=graph.graph.edges[source,target,'PAPER_CITES_PAPER'];origin=a.get('observations',{}).get(source,{})
            if origin.get('metadata_source') not in OPENALEX_SOURCES:continue
            p=controller.papers[source];t=controller.papers.get(target)
            record={'source':canonical_id(source),'target':canonical_id(target),'relation':'PAPER_CITES_PAPER',
                'direction':'source_cites_target','source_title':(p.title or '')[:24],'source_year':p.year,
                'target_state':'retrieved' if t else 'unresolved',
                'provenance':{'metadata_source':origin['metadata_source'],'field':'referenced_works','observed_at':origin['retrieved_at']}}
            if t:record.update(target_title=(t.title or '')[:24],target_year=t.year)
            records.append(record)
        groups.append(records)
    # Round-robin high-relevance anchors, with canonical-ID tie breaking.
    eligible=[g[i] for i in range(max(map(len,groups),default=0)) for g in groups if i<len(g)]
    return eligible,citations.number_of_edges()


def pack(controller):
    common=controller._common_context();baseline=deepcopy(common)
    eligible,available=available_evidence(controller)
    quota=min(MAX_GRAPH_TOKENS,max(0,(controller.planner.limits.max_active_tokens or 4096)//4))
    chosen=[]
    for edge in eligible[:MAX_EDGES]:
        if count(serialized([*chosen,edge]))<=quota:chosen.append(edge)
    context=deepcopy(common)
    context.update(graph=controller.graph.stats(),candidates=score_candidates(controller.question,controller.papers,controller.graph,controller.scoring)[:8],
        subgraph=chosen,topic_coverage=dict(Counter(t.id for p in controller.papers.values() for t in p.topics).most_common(8)))
    def exceeds():
        return (len(serialized(context))>controller.budget.max_context_chars-1024 or
            count(serialized(controller.planner._parts(context)[1]))>(controller.planner.limits.max_active_tokens or 10**12)-64)
    displaced=Counter()
    # Keep basic scholarly evidence/history; edges have a small explicit share,
    # rather than being the first field automatically discarded.
    floors=[('candidates',4),('recent_actions',1),('observed_metadata',2),('top_papers',2),('candidates',0)]
    while exceeds():
        for field,minimum in floors:
            if len(context[field])>minimum:
                context[field].pop();displaced[field]+=1;break
        else:
            if len(context['topic_coverage'])>3:
                context['topic_coverage'].pop(next(reversed(context['topic_coverage'])));displaced['topic_coverage']+=1
            elif len(context['subgraph'])>1:
                context['subgraph'].pop();displaced['graph_edges_capacity']+=1
            else:
                # Tiny/unusual envelopes may not admit even one full edge.
                # Preserve the existing minimum common context; report absence.
                if context['subgraph']:context['subgraph'].pop();displaced['graph_edges_capacity']+=1
                else:
                    context=baseline;context.update(graph=controller.graph.stats(),candidates=[],subgraph=[],topic_coverage={})
                    if exceeds():raise ValueError('Minimum common graph envelope exceeds capacity')
                    break
    selected=context['subgraph']
    before_tokens=count(serialized(baseline));after_common={k:v for k,v in context.items() if k not in ('graph','candidates','subgraph','topic_coverage')}
    allocation={'policy':VERSION,'available_edges':available,'eligible_edges':len(eligible),'selection_considered_edges':min(len(eligible),MAX_EDGES),
        'selected_edges':len(selected),'excluded_by_capacity':min(len(eligible),MAX_EDGES)-len(selected),
        'excluded_by_neighborhood_cap':max(0,len(eligible)-MAX_EDGES),'graph_token_quota':quota,'selected_graph_tokens':count(serialized(selected)) if selected else 0,
        'selected_record_sha256':[identity(e) for e in selected],
        'eligible_identity_sha256':hashlib.sha256(serialized([(e['source'],e['target']) for e in eligible]).encode()).hexdigest(),
        'displaced_records':dict(displaced),'displaced_common_study_tokens':max(0,before_tokens-count(serialized(after_common))),
        'metadata_titles_are_prefixes':True,'hidden_target_metadata_accessed':False,
        'selection_reason':'relevance-ranked observed anchors; round-robin verified outgoing references; canonical ties'}
    controller._graph_registry.update({identity(e):e for e in selected})
    controller._graph_allocation=allocation
    controller._log('graph_evidence_allocation',allocation)
    return context


def extract_edges(value):
    """Recognize complete structured records, including complete archive slices."""
    result=[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('relation')=='PAPER_CITES_PAPER' and isinstance(x.get('source'),str) and isinstance(x.get('target'),str):result.append(x);return
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str):
            # Active memory and archive chunks can contain JSON lines/fragments.
            decoder=json.JSONDecoder();i=0
            while i<len(x):
                start=x.find('{',i)
                if start<0:break
                try:obj,n=decoder.raw_decode(x[start:])
                except ValueError:i=start+1;continue
                walk(obj);i=start+n
    walk(value)
    return list({identity(e):e for e in result}.values())


def audit_wire(raw, state):
    """Observe actual HTTP request bytes, not merely a planned prompt section."""
    body=json.loads(raw);payloads=[]
    for message in body.get('messages',[]):
        if message.get('role')=='user':
            try:payloads.append(json.loads(message['content']))
            except (ValueError,TypeError):pass
    records=extract_edges(payloads)
    registry=(state or {}).get('registry',{})
    verified=[e for e in records if identity(e) in registry and registry[identity(e)]==e]
    allocation=deepcopy((state or {}).get('allocation',{}))
    pairs={(e['source'],e['target']) for e in verified}
    selected_hashes=set(allocation.get('selected_record_sha256',[]))
    current_pairs={(e['source'],e['target']) for e in verified if identity(e) in selected_hashes}
    return {**allocation,'boundary':'actual-httpx-request-content','wire_sha256':hashlib.sha256(raw).hexdigest(),
        'wire_bytes':len(raw),'serialized_request_study_tokens':count(raw.decode()),
        'delivered_edges':len(pairs),'delivered_record_versions':len(verified),'edge_count_definition':'Distinct canonical source-target pairs per serialized request; token counts include all complete verified versions','unverified_edge_records':len(records)-len(verified),
        'delivered_graph_tokens':count(serialized(verified)) if verified else 0,
        'token_measurement_definition':'Study tokenizer on complete verified edge records; not provider billing attribution',
        'delivered_edge_identities':sorted(pairs),
        'not_delivered_after_selection':len(set(allocation.get('selected_record_sha256',[]))-{identity(e) for e in verified}),
        'delivered_current_selection_edges':len(current_pairs),
        'delivered_historical_edges':len(pairs-current_pairs),
        'delivered_historical_record_versions':len({identity(e) for e in verified}-selected_hashes),
        'admission_note':'CLM may keep evidence pending or remove it; no out-of-band graph injection',
        'architecture':(state or {}).get('architecture'),'model_managed_context':(state or {}).get('mode')=='clm'}
