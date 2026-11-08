# Local configuration and hosted transport

Copy `.env.example` to `.env` on your own machine. Keep actual credentials and
private endpoints out of version control. No keys are needed for offline tests,
fixed-corpus deterministic runs or dashboard Diagnostics.

OpenAlex settings configure your key, optional contact email, database, request
spacing, timeout/retries, HTTP-attempt limit and observed cost stopping threshold.
A fixed-corpus run does not have live OpenAlex access.

Hosted planning requires `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. The endpoint
must speak the supported HTTPS chat-completions JSON protocol. The supplied
account/model must support the configured JSON mode and generation parameters.
Configure timeout/output limit, optional temperature (`omit` leaves it out),
optional supported reasoning effort, and per-million token prices. Blank or zero
prices mean unconfigured/unknown cost, not free usage. Inspect the hosted adapter
and your provider documentation before enabling optional parameters.

The hosted adapter makes one request per invocation, with classified errors and
no hidden paid retries. It validates parsed closed schemas and rejects malformed
output. Fallback events, reported usage and estimated unreported usage stay
separate. Dashboard hosted runs require explicit cost confirmation and empirical
readiness. Unknown prices cannot enforce a meaningful dollar ceiling.

For locally reserved questions, maintain a private list of objects with a
`question` field at `data/study/reserved-questions.json`. The individual UI checks
that optional local list. The public release cannot know an absent private holdout
set. Do not use an unconfigured guard as evidence of independence or approval.
No reserved question text or hash list is shipped in this release.

For the seven-method Terra path, the existing hosted settings select the Responses
transport. Gemini ADC is a separate optional adapter with an isolated SDK; it is
not configured solely by the chat-completions endpoint variables. See
[seven-method integration](SEVEN_ARCHITECTURES.md).
