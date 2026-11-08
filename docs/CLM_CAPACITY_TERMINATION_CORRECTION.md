# November 8, 2026: legacy CLM capacity and termination

Policy: `controlled-context-v2-capacity-stop` (`clm` and `clm_graph`).

The raw working file and compact serialized user payload each have a configured
character ceiling. Serialization includes protected state, escaping and controls.
Active-plus-recovery and full-input token ceilings apply independently. No limit
was removed or increased. Meta uses a different retained-conversation character
scope; silently equalizing it would change the experimental treatment.

Previously, pending admission could reject a valid STOP and enter controller
acquisition fallback. Now a closed, schema-valid, permitted STOP with required
valid evidence can terminate with pending observations archived but unadmitted.
It does not silently discard evidence or claim unseen text was delivered.
A pending acquisition instead receives context-management feedback within the
existing model-call/token budget. Successful full-batch admission requires a
fresh decision; the blocked acquisition is never automatically replayed.
Current-payload capacity failure or failed environment admission commit takes
the existing resource-stop path, preventing unrelated acquisition fallback.
Ordinary invalid scholarly-action handling remains separate.

New receipts distinguish admission, pending management, capacity failure and
model STOP with or without pending observations. Result metadata records pending
indices and context-policy identity; controlled manifests record
`legacy_context_policy_version`. Bounded operational feedback may be omitted
when necessary without dropping evidence or relaxing a request limit.

Synthetic public regressions cover boundaries, escaping, token guards, STOP,
invalid edits/recovery, bounded repeated management, provenance, failed admission
and zero acquisition fallback. Exact historical eight-case request replays remain
private rather than distributing generated research traces. Meta and conventional
regressions test their existing contracts independently.

This changes realized legacy CLM control flow and possibly trajectory lengths,
call usage and fallback counts. New experiments require fresh source pins and a
new protocol version; old results must not be relabeled or pooled as corrected.
Offline tests establish no improvement in retrieval, factual grounding or cost.
Graph exposure, task framing and independent judgments remain scientific gates.
