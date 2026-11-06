# CLM-inspired working-context adaptation

The conventional controller selects a bounded working view from observed state.
The CLM controller permits model-requested, schema-validated text edits to
`active_context.txt`. A pinned envelope retains the question, policy, tool schema
and budgets. Raw observations and graph/metadata records remain authoritative.

Both architectures can recover archived observations under the same operational
policy. Requested edits, committed edits, rejection/failure and subsequent actual
inputs are recorded. An edit counts as applied only after its commit succeeds.
A failed or uncertain commit stops use of that context; it must not appear as a
successful synchronization event. Management/recovery calls consume the common
inference budget and are reported separately from retrieval actions.

The pressure diagnostic accepts horizon, evidence position, distractors,
correction, active capacity and archive availability. It measures delivered
study tokens and distinct source text against capacity, validates delivery and
completion, and retains invalid conditions for audit while excluding them from
aggregate displays. Audit storage is distinct from operational recovery access.
Retention, initial-probe recovery and correction accuracy are distinct metrics.

The bundled tokenizer measures explicit study inputs, not undocumented native
provider framing. Context inspection reports observable requests/transitions,
not hidden reasoning. Scripted diagnostic policies are engineered correctness
fixtures, not evidence of learned context-management skill.

This adaptation provides no model training, reinforcement learning, unrestricted
shell, arbitrary filesystem access, full chat-role rewriting, provider-native
context-window control, or compute-cache reuse. It does not reproduce another
project's training harness or establish equivalent published benchmark scores.
No superiority claim follows from a successful edit or a synthetic retention run.
