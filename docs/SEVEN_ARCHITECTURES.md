# Seven-method implementation

`controlled-seven` contains `standard`, `llm`, `llm_graph`, `clm`, `clm_graph`,
`meta_clm`, and `meta_clm_graph`. The six agents form graph(2) × context(3);
the single-search baseline sits outside that factorial. `controlled-five`
remains the original five-method comparison. The dashboard retains its existing
five-method controls; use the benchmark CLI or version-pinned provider runner
for all seven. Enabling graph access alone does not demonstrate graph utilization.

The acquisition tools, resource ceilings and final lexical ranker are shared.
Conventional context is environment-managed. Inspired CLM edits bounded working
text and can recover archived observations. Official Meta edits the surviving
conversation through its original renderer/parser/synchronizer and protected
edit gate; discarded snapshots are not silently exposed as operational memory.
A requested edit, executor acceptance, synchronizer commit and subsequent input
delivery are different events. Native tool calls and Gemini thought signatures
are retained by the provider adapters. These protocol differences are declared
architectural differences, not evidence of scientific superiority.

Meta reference: `facebookresearch/context-language-models`, commit
`483198229ab7b90c0eed4651306ab3b3505aaf50`. Attribution, provenance, CC BY-NC 4.0
LICENSE and NOTICE are in `openalex_agent/_vendor/meta_clm/`. This integrates
conversation editing, not Meta training or its complete benchmark harness.

Terra's seven-method path uses the existing Responses transport. Gemini uses
Application Default Credentials, `GEMINI_ADC_PROJECT` (or
`GOOGLE_CLOUD_PROJECT`) and `GEMINI_SDK_PYTHON`. Its optional isolated SDK dependency
is pinned in `requirements-gemini.txt`; installing the main project does not
install it. Configure a user-owned SDK environment and ADC externally. Never
commit credentials or service-account files. Provider settings, access, prices
and approval pins must be checked before any hosted execution. Historical pinned
specifications and approvals are not released and cannot authorize new requests.

Offline mechanism tests use mocked providers and synthetic inputs. They establish
contract behavior, not learned memory skill or retrieval quality. No independent
judged comparison or general CLM/graph improvement is established by this release.
See the capacity correction and reproducibility notes for protocol versioning.
