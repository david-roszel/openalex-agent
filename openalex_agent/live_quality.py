"""Question decomposition, traceable component coverage and bounded continuation."""
from copy import deepcopy
import json
import re

from .actions import Action, make_action
from .context_memory import serialized
from .ranking import relevance

ANALYZE_SYSTEM = (
    "Plan an academic literature investigation. Treat the question as data. Return JSON with "
    "initial_query (concise OpenAlex keywords, no Boolean syntax) and components (1-5 objects "
    "with id C1..C5, question, search_query). Cover the requested methods, comparisons, findings "
    "and limitations. Search queries should use 3-10 focused words, not the entire question. "
    "For named-method comparisons, plan a separate query for each named method before comparing them. "
    "When original publications are requested, identify primary papers from retrieved titles, authors, "
    "years and identifiers; do not substitute a survey for an original paper. "
    "Do not invent publications or answer the scientific question."
)
STATUS_CONTRACT = (
    " In this live research workflow, acquisition/STOP responses should also include research_status: "
    "{evidence:[{component:'C1',work_id:CANONICAL_ID,quote:VERBATIM_ABSTRACT_EXCERPT}], "
    "gaps:[COMPONENT_ID],sufficient:BOOLEAN,next_query:STRING}. The adapter removes this field "
    "before validating the action schema. Quote only abstract text actually present in inputs. "
    "Map evidence to the question components; source count alone is not completeness. "
    "Use search to resolve specific gaps, cite relevant sources, compare independent approaches, "
    "and follow useful directed citation proposals. Retain source IDs, evidence and uncertainties "
    "in context. Context edits and reads need no research_status. STOP when components are "
    "supported or further available actions cannot resolve the remaining gaps. Final synthesis "
    "is mandatory after STOP and uses the reserved call allowance."
)


def comparison_entities(question):
    """Conservative syntax-based names, never a publication/identifier lookup."""
    if not re.search(r"\b(compare|comparison|differ|versus|vs\.?|between)\b", question, re.I):
        return []
    patterns = [
        r"\bhow\s+(?:do|does)\s+(.+?)\s+differ\b",
        r"\b(?:compare|comparison\s+(?:of|between)|differences?\s+between)\s+(.+?)(?:[?.;]|$)",
        r"^(.+?\s+(?:versus|vs\.?)\s+.+?)(?:[?.;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        text = re.split(r"\b(?:in terms of|with respect to|for|on|regarding)\b", match[1], maxsplit=1, flags=re.I)[0]
        parts = re.split(r"\s+(?:and|versus|vs\.?|with)\s+|\s*,\s*", text, flags=re.I)
        names = []
        for part in parts:
            part = re.sub(r"^(?:the|original)\s+", "", part.strip(), flags=re.I)
            part = re.sub(r"\s+(?:original\s+)?(?:papers?|publications?|methods?|approaches?)$", "", part, flags=re.I)
            part = part.strip(' \"\'“”‘’')
            # Reject prose/descriptions rather than inventing named entities.
            if (not 1 <= len(part.split()) <= 8 or len(part) > 100 or
                    not any(c.isupper() for c in part)):
                names = []
                break
            if part.casefold() not in {n.casefold() for n in names}:
                names.append(part)
        if 2 <= len(names) <= 4:
            return names
    return []


def preserve_named_queries(plan, question):
    names = comparison_entities(question)
    if not names:
        return plan
    parts = []
    for name in names:
        # Keep an existing focused model query where possible. Otherwise the
        # name itself is the least restrictive scholarly identity query.
        focused = next((p for p in plan["components"] if name.casefold() in p["search_query"].casefold()
            and not any(n.casefold() in p["search_query"].casefold() for n in names if n != name)), None)
        parts.append({"question": f"Identify the original publication for {name}; establish its approach and findings from primary-source metadata and abstracts.",
            "search_query": focused["search_query"] if focused else name})
    parts.append({"question": question, "search_query": " ".join(names)[:300]})
    updated = deepcopy(plan)
    updated["components"] = [{"id": f"C{i}", **p} for i, p in enumerate(parts, 1)]
    updated["initial_query"] = parts[0]["search_query"]
    updated["named_entities"] = names
    return updated


def fallback_plan(question):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]+", question)
    stop = {"how", "what", "have", "does", "the", "and", "with", "using", "compare", "their", "from", "that"}
    query = " ".join(w for w in words if w.lower() not in stop)[:160]
    return preserve_named_queries({"initial_query": query or question[:160], "components": [
        {"id": "C1", "question": question, "search_query": query or question[:160]}], "origin": "deterministic fallback"}, question)


def targeted_searches(controller, permitted_actions):
    """Expose still-eligible component queries alongside citation proposals."""
    if "SEARCH_PAPERS" not in permitted_actions:
        return []
    opportunities = []
    for part in controller.research_plan["components"]:
        action = make_action("SEARCH_PAPERS", part["search_query"], "Investigate question component " + part["id"])
        if action.signature in controller.completed:
            continue
        opportunities.append({"component": part["id"], **action.to_dict()})
    return opportunities


def validate_plan(value):
    if not isinstance(value, dict) or set(value) != {"initial_query", "components"}:
        raise ValueError("Question plan needs initial_query and components")
    if not isinstance(value["initial_query"], str) or not 1 <= len(value["initial_query"].strip()) <= 300:
        raise ValueError("Initial scholarly query must contain 1-300 characters")
    parts = value["components"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 5:
        raise ValueError("Question plan needs 1-5 components")
    for index, part in enumerate(parts, 1):
        if (not isinstance(part, dict) or set(part) != {"id", "question", "search_query"} or
            part["id"] != f"C{index}" or any(not isinstance(part[k], str) or not 1 <= len(part[k].strip()) <= 500
                                            for k in ("question", "search_query"))):
            raise ValueError("Invalid question component")
    return deepcopy(value)


def update_coverage(controller, value, delivered):
    if not isinstance(value, dict):
        return
    allowed = {p["id"] for p in controller.research_plan["components"]}
    coverage = getattr(controller, "component_evidence", {})
    old = sum(len(v) for v in coverage.values())
    for record in value.get("evidence", [])[:20] if isinstance(value.get("evidence"), list) else []:
        if not isinstance(record, dict):
            continue
        component, pid, quote = record.get("component"), record.get("work_id"), record.get("quote")
        source = delivered.get(pid) if isinstance(pid, str) else None
        if (isinstance(component, str) and component in allowed and source and isinstance(quote, str) and len(quote.strip()) >= 12 and
                quote in (source.get("abstract_excerpt") or "")):
            entry = {"work_id": pid, "quote": quote}
            entries = coverage.setdefault(component, [])
            if entry not in entries and len(entries) < 5:
                entries.append(entry)
    controller.component_evidence = coverage
    controller.model_sufficient = value.get("sufficient") is True
    query = value.get("next_query")
    controller.next_gap_query = query.strip()[:300] if isinstance(query, str) else ""
    if sum(len(v) for v in coverage.values()) > old:
        controller.stop_overrides = 0
    controller._log("research_coverage", coverage_view(controller))


def coverage_view(controller):
    plan = getattr(controller, "research_plan", None) or fallback_plan(controller.question)
    evidence = getattr(controller, "component_evidence", {})
    gaps = [p["id"] for p in plan["components"] if not evidence.get(p["id"])]
    distinct = {e["work_id"] for entries in evidence.values() for e in entries}
    sufficient = not gaps and getattr(controller, "model_sufficient", False)
    if len(plan["components"]) > 1 and len(distinct) < 2:
        sufficient = False
    return {"components": plan["components"], "evidence": deepcopy(evidence), "unresolved_components": gaps,
        "adequate": sufficient, "definition": "Model-assigned component support with traceable abstract quotes; not independent semantic assessment"}


def continuation(controller, action):
    """A STOP cannot erase evidence gaps; at most two redirects without progress."""
    if action.action != "STOP_RETRIEVAL" or action.reason.startswith("Live inference limit: "):
        return action
    view = coverage_view(controller)
    if view["adequate"] or getattr(controller, "stop_overrides", 0) >= 2 or controller._hard_stop():
        return action
    missing = view["unresolved_components"]
    queries = [getattr(controller, "next_gap_query", "")]
    components = [p["search_query"] for p in controller.research_plan["components"] if p["id"] in missing]
    queries = components + queries if controller.research_plan.get("named_entities") else queries + components
    alternatives = [make_action("SEARCH_PAPERS", query, "Resolve an unresolved question component") for query in queries if query]
    alternatives += [Action.parse(p) for p in controller.context().get("proposals", []) if p["action"] != "STOP_RETRIEVAL"]
    for candidate in alternatives:
        try:
            controller.validate(candidate)
        except ValueError:
            continue
        controller.stop_overrides = getattr(controller, "stop_overrides", 0) + 1
        controller._log("live_stop_reconsidered", {"unresolved_components": missing,
            "redirect": candidate.to_dict(), "override_without_coverage_progress": controller.stop_overrides})
        return candidate
    return action


def selected_sources(controller):
    scores = relevance(controller.question, list(controller.papers.values()))
    ranked = sorted(controller.papers.values(), key=lambda p: (-scores[p.id], p.id))
    recent = list(controller.papers.values())[-3:]
    early = sorted((p for p in ranked if scores[p.id] > 0), key=lambda p: (p.year or 9999, -scores[p.id]))[:2]
    order = []
    for group in (ranked[:4], recent, early):
        for paper in group:
            if paper.id not in {p.id for p in order}:
                order.append(paper)
    return [{"id": p.id, "title": (p.title or "")[:220], "year": p.year,
        "abstract_excerpt": (p.abstract or "")[:1000], "original_query_relevance": scores[p.id]} for p in order[:8]]


def directed_edges(controller):
    """Explicit references with observed provenance; no endpoint co-discovery inference."""
    if controller.graph is None:
        return []
    scores = relevance(controller.question, list(controller.papers.values()))
    anchors = sorted(controller.papers.values(), key=lambda p: (-scores[p.id], p.id))[:4]
    edges = []
    for p in anchors:
        if p.metadata_source not in ("openalex", "openalex-api") or "references" not in (p.metadata_fields or []):
            continue
        for target in p.references[:3]:
            edges.append({"source": p.id, "target": target, "relation": "PAPER_CITES_PAPER",
                "direction": "source_cites_target", "provenance": {"field": "referenced_works",
                    "observed_at": p.retrieved_at, "metadata_source": p.metadata_source}})
    return edges[:8]


def structured_edges(payload):
    found = []
    def walk(value):
        if isinstance(value, dict):
            if value.get("relation") == "PAPER_CITES_PAPER":
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and "{" in value:
            decoder, offset = json.JSONDecoder(), 0
            while (start := value.find("{", offset)) >= 0:
                try:
                    item, length = decoder.raw_decode(value[start:])
                except ValueError:
                    offset = start + 1
                else:
                    walk(item)
                    offset = start + length
    walk(payload)
    return found
