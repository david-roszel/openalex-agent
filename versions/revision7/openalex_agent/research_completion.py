"""Opt-in, versioned shared research objective; no memory or retrieval policy."""

VERSION = "research-completion-guidance-v1"
GUIDANCE = (
    "Research completion guidance: Retrieve when an available scholarly action could "
    "resolve an identifiable evidence gap in the research question. Choose "
    "STOP_RETRIEVAL when the retrieved evidence is sufficient to address the question, "
    "or further available retrieval is unlikely to help. Neither retrieval nor "
    "context edits are required."
)
SUFFIX = "\n\n" + GUIDANCE


def clarify(system):
    """Append the same neutral text without rewriting method-specific instructions."""
    return system + SUFFIX
