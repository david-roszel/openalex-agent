"""Saved five-method evaluation workspace. Presentation and local reads only."""
from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st

from .controlled import ARCHITECTURES
from .evaluation_artifacts import discover, load_evaluation, summary_row, timeline, number
from .research import redact, graph_view
from .ui_theme import METHOD_COLORS, method_colors, METHOD_NAMES, MISSING, card, chart_style, format_value


@st.cache_data(ttl=15,show_spinner=False)
def saved_view(directory,root,assessment):
    return load_evaluation(directory,root,assessment or None)


def table(rows):
    frame=pd.DataFrame(rows)
    for col in frame:
        frame[col]=frame[col].map(lambda v:format_value(v) if v is None or isinstance(v,float) and pd.isna(v) else v)
        if frame[col].map(lambda x:isinstance(x,str) and x==MISSING).any():frame[col]=frame[col].map(str)
    return frame


def _bar(summaries,metric,title):
    rows=[dict(method=r['method'],architecture=METHOD_NAMES[r['method']],value=r.get(metric))
          for r in summaries if r.get(metric) is not None]
    if not rows:st.info(title+': '+MISSING);return
    chart=alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4,size=25).encode(
        y=alt.Y('architecture:N',sort=[METHOD_NAMES[m] for m in ARCHITECTURES],title=None),
        x=alt.X('value:Q',title=title),
        color=alt.Color('method:N',scale=alt.Scale(domain=list(method_colors()),range=list(method_colors().values())),legend=None),
        tooltip=['architecture:N',alt.Tooltip('value:Q',title=title)])
    st.altair_chart(chart_style(chart).properties(height=190),width='stretch')


def _timeline(rows,secrets):
    points=timeline(rows)
    if not points:st.info('No recorded timeline events. '+MISSING);return
    families=st.multiselect('Event types',['Retrieval','Planning','Context','Failure','Control'],
        default=['Retrieval','Planning','Context','Failure'],key='evaluation-event-types')
    filtered=[p for p in points if p['family'] in families]
    if not filtered:st.info('No events match the selected types.');return
    frame=pd.DataFrame(redact([{k:v for k,v in p.items() if k!='record'} for p in filtered],secrets))
    brush=alt.selection_interval(encodings=['x'])
    chart=alt.Chart(frame).mark_circle(size=65).encode(
        x=alt.X('sequence:Q',title='Recorded event sequence within each trajectory'),
        y=alt.Y('method:N',sort=list(ARCHITECTURES),title=None),
        color=alt.Color('method:N',scale=alt.Scale(domain=list(method_colors()),range=list(method_colors().values())),title='Architecture'),
        shape=alt.Shape('family:N',title='Event type'),opacity=alt.condition(brush,alt.value(1),alt.value(.25)),
        tooltip=['method:N','sequence:Q','kind:N','action:N','timestamp:N']).add_params(brush)
    st.altair_chart(chart_style(chart).properties(height=220),width='stretch')
    st.caption('Drag to highlight a range. Sequence numbers align local execution order, not a shared wall clock.')
    event=st.selectbox('Inspect timeline event',filtered,format_func=lambda e:redact(f"{METHOD_NAMES[e['method']]} / {e['sequence']} / {e['kind']}",secrets),key='evaluation-event')
    st.json(redact(event['record'],secrets),expanded=False)


def _context(row,secrets):
    detail=row['detail'];records=detail['requests']
    if row['method']=='standard':st.info('Standard search has no LLM context or factual-response call.');return
    if not records:st.info('No recorded model inputs. '+MISSING);return
    st.caption('Actual saved inputs and committed transitions. Hidden model reasoning is not reconstructed.')
    sizes=pd.DataFrame([dict(call=r['call'],active_tokens=r.get('active_tokens'),full_input_tokens=r.get('full_input_tokens')) for r in records])
    sizes=sizes.melt(id_vars='call',var_name='measurement',value_name='tokens').dropna(subset=['tokens'])
    if not sizes.empty:
        chart=alt.Chart(sizes).mark_line(point=True).encode(x=alt.X('call:O',title='Model call'),
            y=alt.Y('tokens:Q',title='Recorded study tokens'),color=alt.Color('measurement:N',
            scale=alt.Scale(range=[method_colors()['llm'],method_colors()['clm']])),tooltip=['call:O','measurement:N','tokens:Q'])
        st.altair_chart(chart_style(chart).properties(height=210),width='stretch')
    selected=st.select_slider('Recorded model call',options=list(range(len(records))),
        format_func=lambda i:str(records[i]['call']),key='evaluation-context-call')
    request=records[selected]
    st.code(redact(request['active'],secrets),language='text',height=240)
    if request.get('recovery'):st.json(redact(request['recovery'],secrets),expanded=False)
    st.caption('Exact evidence-retention quality: '+MISSING+'. A matching quote alone does not establish support or a complete memory test.')
    with st.expander('Context operations and durable commit receipts'):
        if detail['transitions']:
            st.dataframe(table(redact([{k:t.get(k) for k in ('index','source','requested','validated','committed','status','error','before_chars','after_chars')} for t in detail['transitions']],secrets)),hide_index=True,width='stretch')
        else:st.info('No transition receipts available.')
    with st.expander('Full saved system, input and parsed output'):
        st.json(redact(request['request'],secrets),expanded=False)


def _graph(row,root,secrets):
    detail=row['detail'];delivery=detail['graph_delivery']
    if not row['method'].endswith('_graph'):
        st.info('Graph memory is disabled for this architecture; scholarly retrieval capabilities remain available.');return
    st.dataframe(table([{'Structural snapshot':k,'Recorded count':v} for k,v in detail['graph_snapshot'].items()]),hide_index=True,width='stretch')
    st.dataframe(table([{'Actual input evidence':label,'Request snapshots':delivery[key]} for label,key in (
        ('Structured graph information','structured_graph_calls'),('Candidate scores','candidate_calls'),
        ('Citation edges','citation_edge_calls'),('Action target also in delivered candidates','action_target_overlap_calls'))]),hide_index=True,width='stretch')
    st.caption('An enabled graph is not demonstrated model reliance. Target overlap is observational, not causal attribution. Free-text CLM summaries may contain graph information this structured inspection does not parse.')
    if not delivery['structured_graph_calls']:st.warning('No nonempty structured graph payload was identified in the saved model inputs.')
    with st.expander('Delivered graph payloads and associated decisions'):
        st.json(redact(delivery['examples'],secrets),expanded=False)
    result=detail['result'];directory=detail['directory'];ids=result.get('inspected_ids',[])
    if not ids or not directory or not (Path(directory)/'graph.json').is_file():
        st.info('No bounded graph snapshot is available.');return
    seed=st.selectbox('Graph neighborhood seed',ids,key='evaluation-graph-seed')
    cap=st.slider('Graph node limit',5,40,20,key='evaluation-graph-limit')
    try:
        view=graph_view(directory,seed,max_nodes=cap,hops=1)
        from .web_ui import _draw_graph
        _draw_graph(view)
        st.caption('Forest papers, sand unresolved targets, clay authors, olive topics and eucalyptus institutions. Stored relationships do not establish relevance.')
    except (OSError,ValueError,KeyError,TypeError):st.warning('Graph checkpoint is incomplete or cannot be validated.')


def _rankings(row,secrets):
    ranked=row['detail']['rankings'];display=[]
    for rank,item in enumerate(ranked,1):
        p=item.get('paper',{})
        display.append(dict(rank=rank,title=p.get('title'),openalex_id=p.get('id'),year=p.get('year'),
            lexical_score=item.get('score'),provenance=p.get('provenance'),source_url=p.get('source_url')))
    if not display:st.info('No recorded final ranking. '+MISSING);return
    st.caption('Recorded final lexical rankings. Ranking score is not an independent relevance grade.')
    st.dataframe(table(redact(display,secrets)),hide_index=True,width='stretch')
    selected=st.selectbox('Inspect ranked paper',list(range(len(ranked))),format_func=lambda i:redact(f"{i+1}. {ranked[i].get('paper',{}).get('title','Title unavailable')}",secrets),key='evaluation-paper')
    paper=ranked[selected];metadata=paper.get('paper',{})
    st.text(redact(metadata.get('abstract') or 'Abstract unavailable',secrets))
    with st.expander('Original recorded paper, discovery path and provenance'):
        st.json(redact(paper,secrets),expanded=False)


def render_evaluation(root,settings):
    st.title('Scientific evaluation')
    st.caption('Five architectures. Observable research processes. Independently judged outcomes stay separate.')
    secrets=tuple(v for v in (settings.api_key,settings.llm_api_key,settings.email) if v)
    paths=discover(root)
    if not paths:
        st.info('No saved experiments yet. This workspace only reads existing artifacts; it never starts research.')
        st.markdown('Open **Experiment Dashboard** for offline diagnostics, or place an existing saved development benchmark under `data/`.')
        st.caption('Primary methods: standard, llm, llm_graph, clm, clm_graph. Official Meta CLM remains exploratory.')
        return
    a,b=st.columns([5,1],vertical_alignment='bottom')
    default=next((i for i,p in enumerate(paths) if 'gemini-development' in p),0)
    with a:chosen=st.selectbox('Saved evaluation experiment',paths,index=default,format_func=lambda p:redact(p,secrets),key='evaluation-source')
    with b:
        if st.button('Refresh saved results',width='stretch'):saved_view.clear();st.rerun()
    with st.expander('Optional recorded pooled assessment'):
        st.caption('Select an existing freeze containing raw-output, then a matching scored-output directory. This only reads recorded scores; no scoring or approval is performed.')
        assessment=st.text_input('Scored-output directory under data (optional)',key='evaluation-assessment')
    try:loaded=saved_view(chosen,str(root),assessment)
    except (OSError,ValueError,KeyError,TypeError):st.error('Saved artifacts are outside allowed locations or unavailable.');return
    with st.container(border=True):
        st.markdown('**'+loaded['classification']+'**')
        st.caption(redact('READ ONLY / '+str(loaded['source_status']).upper()+' / '+chosen,secrets))
        if loaded['context_policies']:st.caption('Recorded CLM policy: '+redact(', '.join(loaded['context_policies']),secrets))
        if loaded['synthetic']:st.caption('Synthetic diagnostic data demonstrates software behavior; it is not independently judged research.')
        elif not loaded['relevance_evaluated'] and not loaded['factual_evaluated']:
            st.caption('Relevance, factual accuracy, unsupported-claim rates and retention quality: '+MISSING)
        else:st.caption('Recorded assessment provenance is checked where available. Reviewer independence remains a human attestation, not something this interface proves.')
    for message in loaded['warnings']:st.warning(redact(message,secrets))
    rows=loaded['rows']
    if not rows:
        st.info('No completed comparison rows are available. Missing cells are not scored as zero.')
        for partial in loaded['partial_runs']:
            with st.expander('Partial trajectory / '+str(partial['directory'])):st.json(redact(partial,secrets),expanded=False)
        return
    qids=sorted({str(r['query_id']) for r in rows});trials=sorted({int(r['trial']) for r in rows});budgets=sorted({int(r['action_budget']) for r in rows})
    a,b,c=st.columns([3,1,1])
    with a:qid=st.selectbox('Research question',qids,key='evaluation-question')
    with b:trial=st.selectbox('Trial',trials,format_func=lambda t:f'{t+1} / ID {t}',key='evaluation-trial')
    with c:budget=st.selectbox('Action budget',budgets,key='evaluation-budget')
    question=next((q.get('question') for q in loaded['questions'] if q.get('id')==qid),None)
    if question:st.text(redact(question,secrets))
    selected=[r for r in rows if str(r['query_id'])==qid and int(r['trial'])==trial and int(r['action_budget'])==budget and r.get('variant','full')=='full']
    unique=[]
    for method,col in zip(ARCHITECTURES,st.columns(5,gap='small')):
        cells=[r for r in selected if r['method']==method]
        row=cells[0] if len(cells)==1 else None
        summary=summary_row(row) if row else {}
        if row:unique.append(row)
        with col:card(method,available=bool(row),actions=summary.get('actions'),inspected=summary.get('inspected'),
            state='Recorded' if row and row['detail']['result'] else 'Partial')
        if len(cells)>1:st.warning('Duplicate '+method+' cells are excluded from the comparison; no outcome is selected.')
    st.caption('Matched question, trial and budget. The standard baseline uses one search and no LLM. Missing cells remain unavailable.')
    summaries=[summary_row(r) for r in unique]
    overview,process,memory,rankings,provenance=st.tabs(['Architecture comparison','Research timeline','Graph & context','Paper rankings','Evidence & provenance'])
    with overview:
        options={'Inspected papers':'inspected','Retrieval actions':'actions','Model calls':'model_calls',
            'Reported total tokens':'reported_tokens','Mean model-request latency (s)':'mean_latency','Trajectory runtime (s)':'runtime','Estimated gross API cost (USD)':'cost'}
        metric=st.selectbox('Comparison measurement',list(options),key='evaluation-metric')
        _bar(summaries,options[metric],metric)
        st.dataframe(table([{'Architecture':METHOD_NAMES[r['method']],'Actions':r['actions'],'Inspected':r['inspected'],
            'Model calls':r['model_calls'],'Reported tokens':r['reported_tokens'],'Gross cost (USD)':r['cost'],
            'Mean request latency (s)':r['mean_latency'],'Runtime (s)':r['runtime'],'Fallbacks':r['fallbacks'],'Request/parse failures':r['failures']} for r in summaries]),hide_index=True,width='stretch')
        st.caption('Request latency includes recorded model turns such as context management. Costs are recorded usage-based estimates, not verified invoices or promotional-credit balances.')
        st.subheader('Independent quality measurements')
        st.dataframe(table([{'Architecture':METHOD_NAMES[r['method']],'P@10':r['precision'],
            'Pooled nDCG@10' if loaded.get('pooled_assessment') else 'Recorded nDCG@10':r['ndcg'],
            'Factual accuracy':r['factual_accuracy'],'Unsupported claims':r['unsupported_claims'],
            'Evidence retention':r['retention']} for r in summaries]),hide_index=True,width='stretch')
        st.caption('No labels are inferred from ranks, citations, quotation matches or successful requests. Baseline factual response is not applicable. Unknown usage is not free inference.')
    with process:
        progress=[{**point,'method':r['method']} for r in unique for point in r['detail']['progression'] if number(point.get('inspected')) is not None]
        if progress:
            chart=alt.Chart(pd.DataFrame(progress)).mark_line(point=True).encode(x=alt.X('actions:Q',title='Retrieval actions'),
                y=alt.Y('inspected:Q',title='Inspected papers'),color=alt.Color('method:N',
                scale=alt.Scale(domain=list(method_colors()),range=list(method_colors().values()))),tooltip=['method:N','actions:Q','inspected:Q','new_papers:Q'])
            st.altair_chart(chart_style(chart).properties(height=230),width='stretch')
        _timeline(unique,secrets)
    with memory:
        if unique:
            chosen_row=st.selectbox('Architecture to inspect',unique,format_func=lambda r:METHOD_NAMES[r['method']],key='evaluation-inspect')
            s=summary_row(chosen_row)
            st.dataframe(table([{'Committed model edits':s['committed_edits'],'Recovery attempts':s['recovery_attempts'],
                'Accepted recovery reads':s['accepted_recoveries'],'Retention quality':s['retention'],
                'Context policy':redact(s['context_policy'],secrets)}]),hide_index=True,width='stretch')
            graph,context=st.columns(2,gap='large')
            with graph:st.subheader('Graph evidence');_graph(chosen_row,root,secrets)
            with context:st.subheader('Working context');_context(chosen_row,secrets)
    with rankings:
        if unique:
            chosen_row=st.selectbox('Ranked architecture',unique,format_func=lambda r:METHOD_NAMES[r['method']],key='evaluation-rank-method')
            _rankings(chosen_row,secrets)
    with provenance:
        st.info('Stored abstracts and graph edges can exist without reaching a model request. Exact delivery is distinct from retention, and retention is distinct from correctness.')
        st.caption('This view does not reconstruct absent passages or judge semantic support. Inspect exact inputs, source metadata, errors, stops and versions before interpreting outcomes.')
        st.dataframe(table(redact([{'Architecture':METHOD_NAMES[r['method']],'Stop reason':r['stop_reason'],'Context policy':r['context_policy'],
            'Committed edits':r['committed_edits'],'Accepted recoveries':r['accepted_recoveries'],
            'Structured excerpt sources (lower bound)':r['excerpt_sources'],'Unadmitted characters at termination':r['pending_chars']} for r in summaries],secrets)),hide_index=True,width='stretch')
        st.caption('Excerpt-source counts inspect structured actual inputs only. Free-text summaries may carry additional evidence. Unrecorded terminal pending state is not assumed empty.')
        for row in unique:
            for warning in row['detail']['warnings']:st.warning(METHOD_NAMES[row['method']]+': '+warning)
        with st.expander('Frozen settings and source identity'):st.json(redact(loaded['manifest'],secrets),expanded=False)
        export=table(redact(summaries,secrets))
        for column in export:
            export[column]=export[column].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
        st.download_button('Download displayed observations (CSV)',export.to_csv(index=False),
            file_name='evaluation-observations.csv',mime='text/csv',key='evaluation-download')
        st.caption('Exports contain observed counters and explicit unavailable values. They do not modify any saved artifact.')
