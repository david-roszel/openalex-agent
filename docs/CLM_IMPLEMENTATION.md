# CLM-inspired working context and recovery

The primary CLM implementation exposes a bounded active working text to model-authored edits. It is distinct from the optional official Meta conversation-editing architecture. The wrapper preserves protected system instructions, original question, scope and budgets outside editable content.

`ContextPlanner` sends conventional externally selected observations or the actual current `context/active_context.txt` contents in CLM mode. Model replies may select scholarly actions or request a validated working-context edit, archive listing or bounded observation recovery. The environment, rather than the edit itself, archives authoritative observations. Edits do not grant shell/file-system execution or erase raw source provenance.

Operation receipts distinguish request, validation, commit and later input delivery. A successful edit requires durable replacement/readback; uncertainty blocks inference. Recovery has separate acceptance and later-delivery evidence. It can consume call/context capacity and is not inherently exclusive to the CLM arm. An edit or recovered snippet alone does not prove useful retention.

The November8 legacy correction is identified as **controlled-context-v2-capacity-stop**: limits remain enforced, capacity admission is checked, and a validated STOP with pending archived evidence can terminate. Later R7 source/policies and observed M exposure failures remain separate, not silently relabeled as this correction. See CLM_CAPACITY_TERMINATION_CORRECTION.md and versions/revision7.

The public app additionally provides evidence-aware research planning and terminal synthesis under its own profile. Metadata may be reloaded at synthesis, so a source-traceable final answer is not a pure active-memory test. Actual graph delivery and CLM persistence are recorded, but their causal utility or retention superiority is unestablished.

Optional active/input token ceilings use the packaged checksum-verified study tokenizer; provider billing/framing equivalence is not claimed. Character, context, action, token, call and spending limits remain distinct. No budget increase or memory policy optimization is introduced by this GitHub preparation.

Official Meta uses its original restricted conversation renderer, parser, synchronizer/edit gate and rollback behavior. It is exploratory; see SEVEN_ARCHITECTURES.md and vendored notices. Both mechanisms require source/prompt/provider identities in future experiments, with historical traces left unchanged.
