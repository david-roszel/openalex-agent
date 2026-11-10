# Iterations retained in this update

The public update follows the existing Git history, rather than reinitializing it or combining incompatible experiment policies. The prior published base is9b6d564 (seven-method implementation); c4bc32e is the first public version.

|Development line reviewed|Useful contribution|Public disposition|
|---|---|---|
|Original NewProject|Retrieval/cache/graph core, hosted planning, evaluation and four-workspace UI|Its core is inherited by later source; earlier public behavior remains in Git history. No redundant complete copy is added|
|UI / November8 validation|Forest/sage/tan/cream light/dark styling, alignment, saved-output masking, reliability regressions|Included through the later tested source and focused/all regression tests|
|Unified integration candidate|Reconciles UI and reliability changes|Inherited by current source; no second competing root package|
|Graph-visible V2|Actual directed-edge HTTP-boundary records, provenance/capacity accounting and Terra request holds|Retained unchanged inside the frozen R7 runtime, which includes these modules|
|Multihorizon V3 / runtime revisions1–3|S/L/M design, acquisition/inspection separation, itemwise abstentions, required-final outcomes, resource/exposure diagnostics|Runtime3 is retained separately under versions/revision7. Earlier runtime files are superseded, with the original workspaces preserved locally|
|Revision6 archive|Historical experimental semantics|Not relabeled as corrected R7; no large duplicate archive is copied into Git|
|Current LiveDemo|Question→retrieval→cited synthesis UI, full dashboard, graph/context source delivery, token/accounting/deadline guards, saved-view presentation|Primary root application and regression suite|
|Stopped follow-up and recovery/support work|Evidence of access/accounting blockers, never-run work and limitations|Summarized in results/limitations documentation; no blocked work is called completed|
|Final private grading package|Private credential adapter, duplicated application packaging, course-only directions|Excluded entirely from GitHub, especially grader.env and credential-bearing ZIPs|
|Legacy progress monitor|Read-only queue/WAL snapshot, conservative exposure and terminal/HTML progress|Portable source/tests included under tools/legacy-monitor; only workstation default input/output paths changed. Fixed900-cell historical schema is explicitly limited|
|Standalone results dashboard|Historical campaign-specific statistical/result display|Inspected, but hard-coded monitor dependency and actual-private-data UI tests make it unsuitable as a portable default. Main Scientific Evaluation views and isolated R7 analysis cover the supported release paths; no untested dashboard rewrite or raw output bundle is copied|

A module-name/hash comparison found every original NewProject/UI/integration root module represented in current LiveDemo. The benchmark line adds distinct modules and changes shared controllers, so copying those over the live app would mix experimental definitions. The isolated runtime avoids that.

Existing public portability improvements in development_review.prepare (explicit caller inputs) and freeze_pilot (explicit file mapping) are retained instead of restoring private-development hard-coded defaults. New development-only pooled collection is retained from the current implementation. Frozen R7 source is byte-preserved; only its selected synthetic test helper import is relocated, without changing assertions or scientific scoring.

No experimental results, corpus, private ledgers, approvals or human judgments are changed. New tests establish engineering behavior only. Source hashes for future experiments change; prior experiments keep their original hashes and must not be silently pooled with this release.

Exact frozen Meta license/source EOF spacing and the original protocol's Markdown line breaks are retained. Three narrowly scoped .gitattributes rules suppress only those whitespace warnings; source pins, secret checks and ordinary code whitespace checks remain enforced.
