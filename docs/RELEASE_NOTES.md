# Publication security and documentation cleanup

The consolidated release was committed as `806d673` and merged into main as `f76e842f`. This follow-up addresses the historical security audit's remaining publication warnings without changing retrieval, graph memory, CLM, evaluation scoring, resource limits or frozen R7 source.

Sensitive-file exclusions now cover ADC/OAuth files, private keys, held-out/annotation inputs, local receipts/traces and private notebook/package outputs. The publication auditor also rejects recognizable renamed credential JSON. Portable publication instructions replace workstation-specific commands; the synthetic path-security test uses an anonymous example. Licensing and security guides distinguish original code without a project-wide license, preserved noncommercial Meta components, tokenizer attribution, verified false positives and audit limits.

The five primary architectures remain `standard`, `llm`, `llm_graph`, `clm`, `clm_graph`. Official Meta CLM remains exploratory. No private credentials, corpus, held-out questions, independent judgments, approvals or raw experiment outputs are added. No hosted experiment or scientific-performance claim is introduced.

The November 10 historical audit covered all four reachable commits and locally unreachable objects, with no confirmed secret exposure. Current cleanup checks and actual offline regression results are recorded in [VALIDATION.md](VALIDATION.md). Public visibility, remote CI/review and the owner's original-code license decision are separate actions; this cleanup does not commit, push, merge or change visibility.
