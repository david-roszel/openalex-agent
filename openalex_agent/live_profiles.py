"""Submission defaults shared by the UI, CLI and independent evaluation."""
from dataclasses import replace
from urllib.parse import urlsplit

from .config import Settings
from .context_planner import InferenceLimits
from .controller import Budget

PROFILE_VERSION = "thorough-research-v2"
THOROUGH_BUDGET = Budget(max_actions=18, max_rounds=18, max_inspected=144,
    max_api_requests=40, max_llm_calls=28, page_size=6, output_k=30,
    patience=5, max_context_chars=24000)
THOROUGH_LIMITS = InferenceLimits(max_total_tokens=150000, max_llm_cost_usd=1.50)


def priced_settings(settings):
    """Explicit conservative Standard-tier profile, only for the exact known model.

    Official pricing checked 2026-11-09: $1 input / $5 output per million,
    up to $2.50 input cache writes. No discount is assumed. Other endpoints or
    models require user-supplied positive prices; zero never means free.
    """
    if (urlsplit(settings.llm_base_url).hostname == "api.openai.com" and
            settings.llm_model == "gpt-6.1-terra" and
            settings.llm_input_price == settings.llm_output_price == 0):
        return replace(settings, llm_input_price=2.50, llm_output_price=5.00)
    return settings


def thorough_config(question, *, variant="combined", mode="live", corpus=None, planner="hosted", **changes):
    from .research import ResearchConfig
    if variant not in ("conventional", "combined", "graph", "clm", "standard"):
        raise ValueError("Choose conventional, combined, graph, clm or standard")
    options = dict(question=question, mode=mode, corpus=corpus, planner=planner,
        controlled=True, graph_memory=variant in ("combined", "graph"),
        context_management="clm" if variant in ("combined", "clm") else "conventional",
        budget=THOROUGH_BUDGET, inference_limits=THOROUGH_LIMITS,
        final_synthesis=variant != "standard", quality_aware=True,
        synthesis_revisions=1, max_seconds=600.0)
    if variant == "standard":
        options.update(planner="deterministic", controlled=False, quality_aware=False,
            budget=replace(THOROUGH_BUDGET, max_actions=1, max_rounds=1, page_size=30),
            synthesis_revisions=0)
    if planner == "deterministic":
        options["inference_limits"] = replace(THOROUGH_LIMITS, max_llm_cost_usd=None)
    options.update(changes)
    return ResearchConfig(**options)


def load_live_settings(env_file=None):
    return priced_settings(Settings.from_env(env_file) if env_file else Settings.from_env())
