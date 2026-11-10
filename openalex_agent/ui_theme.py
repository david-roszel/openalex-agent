"""Presentation only: shared nature palette and small, scoped Streamlit styles."""
import html
import math

CREAM = "#F6F1E7"
PINE = "#244B3B"
SAGE = "#8BA58B"
TAN = "#D8C4A5"
OLIVE = "#657A3F"
CHARCOAL = "#29332D"
METHOD_COLORS = {"standard":"#786B57", "llm":PINE, "llm_graph":OLIVE,
                 "clm":"#A7794C", "clm_graph":"#456B6A"}
NODE_COLORS = {"paper":PINE, "author":"#A7794C", "topic":OLIVE,
               "institution":"#456B6A", "unresolved":"#D8C4A5"}
METHOD_NAMES = {"standard":"Standard search", "llm":"LLM", "llm_graph":"LLM + graph",
                "clm":"CLM", "clm_graph":"CLM + graph"}
MISSING = "NOT EVALUATED"


def format_value(value, *, money=False):
    if value is None or isinstance(value,float) and not math.isfinite(value):return MISSING
    if isinstance(value,(float,int)) and not isinstance(value,bool):
        if money:return f"${value:,.4f}"
        return f"{value:,.2f}".rstrip('0').rstrip('.') if isinstance(value,float) else f"{value:,}"
    return str(value)


DARK_METHOD_COLORS = dict(zip(METHOD_COLORS, ["#C6B99E", "#A9C4AC", "#B2C680", "#D6AA7D", "#8FBDB8"]))


def palette(mode=None):
    """Resolve native Streamlit appearance without changing research state."""
    if mode is None:
        import streamlit as st
        mode = st.context.theme.type or st.get_option('theme.base')
    if mode == 'dark':
        return dict(background='#131C18', surface='#1D2A23', text='#EAECE3',
                    muted='#BAC7BA', accent='#A9C4AC', border='#526457', grid='#35473B')
    return dict(background=CREAM, surface='#FFFCF6', text=CHARCOAL,
                muted='#536057', accent=PINE, border=TAN, grid='#E1D8C7')


def method_colors(mode=None):
    return DARK_METHOD_COLORS if palette(mode)['background']=='#131C18' else METHOD_COLORS


def apply_theme():
    import streamlit as st
    # Native TOML theme covers controls/charts; CSS is limited to layout and
    # authored cards. No hiding controls, external fonts or injected scripts.
    st.markdown('''<style>
    .stMainBlockContainer {max-width: 1480px; padding-top: 4.5rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -.035em; font-weight: 600;}
    h2,h3 {letter-spacing: -.02em;}
    [data-testid="stMetric"] {background:light-dark(#FFFCF6,#1D2A23); border:1px solid light-dark(#D8C4A5,#526457);
      border-radius:10px; padding:14px 16px; height:100%;}
    [data-testid="stMetricValue"] {color:light-dark(#244B3B,#A9C4AC); font-variant-numeric:tabular-nums;}
    [data-testid="stDataFrame"] {border:1px solid light-dark(#D8C4A5,#526457); border-radius:8px;}
    [data-testid="stCaptionContainer"] {opacity:1;}
    [data-testid="stCaptionContainer"] p {color:light-dark(#536057,#BAC7BA);}
    .oa-eyebrow {font-size:.72rem; font-weight:700; letter-spacing:.16em;
      text-transform:uppercase; color:light-dark(#52674A,#B2C680); margin-bottom:.5rem;}
    .oa-rule {height:1px; background:light-dark(#D8C4A5,#526457); margin:.4rem 0 1.5rem;}
    .oa-card {border:1px solid light-dark(#D8C4A5,#526457); border-top:4px solid var(--method);
      background:light-dark(#FFFCF6,#1D2A23); border-radius:10px; padding:18px; min-height:264px; box-sizing:border-box; display:grid; grid-template-rows:32px 48px 42px 45px auto; align-content:start;}
    .oa-card-title {font-size:1.08rem; color:light-dark(#244B3B,#A9C4AC); font-weight:650; margin:0; line-height:1.4;}
    .oa-card-sub {color:light-dark(#536057,#BAC7BA); font-size:.79rem; line-height:1.5; margin:0;}
    .oa-chip {display:inline-block; justify-self:start; align-self:start; border:1px solid light-dark(#C4CFB8,#526457); border-radius:4px;
      background:light-dark(#E8EDDF,#293825); padding:3px 7px; font-size:.68rem; letter-spacing:.04em;
      color:light-dark(#244B3B,#A9C4AC); font-weight:650; margin:8px 0;}
    .oa-value {font-size:1.55rem; color:light-dark(#29332D,#EAECE3); font-weight:600; margin-top:9px;}
    .oa-small {font-size:.76rem; color:light-dark(#536057,#BAC7BA); line-height:1.5;}
    @media(max-width:760px) {.stMainBlockContainer {padding:4rem 1rem 2rem;}.oa-card {min-height:0; grid-template-rows:auto auto auto auto auto; gap:6px;}}
    </style>''',unsafe_allow_html=True)


def card(method, *, available, actions=None, inspected=None, state="Recorded"):
    import streamlit as st
    # All dynamic artifact text is escaped; no stored HTML is executed.
    color=method_colors()[method];name=METHOD_NAMES[method]
    description=("Single search · no LLM" if method=='standard' else
        ("Editable context" if method.startswith('clm') else "Selected context")+
        (" · graph enabled" if method.endswith('_graph') else " · no graph"))
    st.markdown(f'''<div class="oa-card" style="--method:{color}">
    <div class="oa-card-title">{name}</div><p class="oa-card-sub">{description}</p>
    <span class="oa-chip">{html.escape(state if available else 'MISSING CELL')}</span>
    <div class="oa-value">{html.escape(format_value(actions))}<span class="oa-small"> actions</span></div>
    <div class="oa-small">{html.escape(format_value(inspected))} inspected papers</div></div>''',unsafe_allow_html=True)


def chart_style(chart):
    # Let the native client theme resolve colors at render time. Python-generated
    # colors can lag a frontend-only appearance switch until the next rerun.
    return chart.configure_view(stroke=None).configure_axis(
        labelFontSize=12,titleFontSize=12)
