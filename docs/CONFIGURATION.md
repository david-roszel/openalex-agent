# Configuration and credentials

Copy `.env.example` to an ignored `.env` only if one does not already exist. Supply your own keys. No working key, instructor credential or private archive is distributed.

The current app loads `.env` relative to its installation, with explicit environment variables taking precedence. Live Research uses the recorded OpenAI-compatible Terra profile; the controlled research backends also preserve other compatible endpoints and optional Gemini/Meta paths. Model access and current pricing require your own account verification. The settings template is a tested code profile, not an API availability guarantee.

|Variables|Purpose|
|---|---|
|LLM_BASE_URL, LLM_API_KEY, LLM_MODEL|Hosted endpoint, private token and model|
|LLM_TIMEOUT, LLM_MAX_OUTPUT_TOKENS|Request/output bounds|
|LLM_TEMPERATURE=omit, LLM_REASONING_EFFORT=low|Recorded provider request settings|
|LLM_INPUT_USD_PER_MILLION, LLM_OUTPUT_USD_PER_MILLION|Conservative accounting rates; zero does not mean free inference|
|OPENALEX_API_KEY, OPENALEX_EMAIL|Separate retrieval access/contact; LLM keys are not OpenAlex credentials|
|OPENALEX_DB|Local cache path, independent from run evidence|
|OPENALEX_REQUEST_TIMEOUT, OPENALEX_MIN_INTERVAL|HTTP pacing/timeout|
|OPENALEX_MAX_REQUESTS, OPENALEX_MAX_RETRIES|Attempt/retry limits; current template retries0|
|OPENALEX_MAX_COST_USD|Observed-response cost guard, not guaranteed prepaid cap|
|OPENALEX_DASHBOARD_PORT|Launcher port, normally8521 simple/8501 full dashboard|
|OPENALEX_DEMO_READ_ONLY|Optional simple-UI recorded-only restriction|

Current default prices are the recorded conservative development bounds2.50 input/5.00 output per million tokens, covering the configured input cache-write case. Actual billing/discounts/credits were not checked for this release preparation. A user-selected per-run ceiling is conditional on correctly configured rates. Unknown provider usage retains reservations; it must not become a free completed request.

Offline tests and diagnostics use mock transports and synthetic settings, with no paid API access. A CLI `--dry-run` prints the effective nonsecret profile, not authentication validity. Do not run connectivity/research commands as a substitute for offline validation unless separately intended and authorized.

Optional Gemini ADC/SDK settings are independent of the direct hosted profile; see SEVEN_ARCHITECTURES.md and requirements-gemini.txt. Never commit application-default credentials, service-account JSON or provider tokens. Frozen R7 source has its own environment/context/policy assumptions; use its isolated instructions.
