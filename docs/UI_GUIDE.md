# UI walkthrough

Install `.[dev,ui]` into the root environment. Application entry points:

|Interface|Launch|URL|Entry|
|---|---|---|---|
|Simple Live Research|`./launch-live-research.sh`|http://127.0.0.1:8521|live_app.py / app.py → live_ui.main|
|Full scientific dashboard|`OPENALEX_DASHBOARD_PORT=8511 ./launch-dashboard.sh`|http://127.0.0.1:8511|legacy_app.py → web_ui.main|

Use separate terminals and different ports to keep both open. Ctrl+C stops each server. They bind localhost; remote use requires a suitable local/SSH port forward, not a public deployment supplied by this repository.

## Live or saved individual research

Enter a question and deliberately click Research for new live work. Configure your own ignored .env first; this can incur costs. Review Advanced Settings: four hosted configurations, execution/source choice and editable per-run limits. Missing credentials disable hosted execution; partial evidence is labeled.

Advanced Settings → Saved research → Open saved research reopens a locally recorded run without starting inference. No original private saved run is checked in. `OPENALEX_DEMO_READ_ONLY=1` optionally restricts the simple UI to recorded viewing. The displayed saved identity/question/settings belong to the recorded result; the form controls configure new runs.

Answers have supporting publication expanders, citation links and limitations. Research trace/resource details show actual input/edit events. The simple UI does not offer a graph canvas or context-diff animation; use the full dashboard or exact local JSON for those records. Source membership/quote checks do not establish independent scientific correctness.

## Full five-workspace dashboard

Scientific Evaluation reads existing saved benchmarks. It offers Architecture comparison, Research timeline, Graph & context, Paper rankings and Evidence & provenance, with question/trial/action-budget selectors. Stored graphs and actual-input delivery are distinct, including cases with an enabled graph but no structured delivered edges.

A fresh public clone contains no private experiments. For a no-API engineering example choose Experiment Dashboard → Scientific category Diagnostics → deterministic planner → Run Comparison. Synthetic/scripted output is explicitly diagnostic, not hosted research effectiveness. The result can be reopened in Saved Experiments/Scientific Evaluation. Long-Horizon Testing also uses scripted diagnostics; it does not establish long-horizon model retention.

Development/Final categories retain reviewed-input, independent-label and paid-consent/approval requirements. Do not substitute diagnostic grades or generated review templates for independent judgments. Missing relevance/grounding/retention labels remain NOT EVALUATED.

Run only source/current app comparisons through this interface. R7 has separate schemas/source policies and is documented under versions/revision7; its entire pilot is not automatically a dashboard menu item. Scientific interpretations are in RESULTS_AND_LIMITATIONS.md.
