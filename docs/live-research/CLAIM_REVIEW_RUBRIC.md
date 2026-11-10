# Independent claim-level review rubric

**Current status: NOT COMPLETED.** No independent human scientific accuracy score has been assigned. `claim-review.json` lists actual generated claims when present; `component-review.json` lists every required component even when no answer has been run. An empty claim list means no new answer to review, not that all claims passed.

Predeclared manageable sample: take Q01 and Q02, both hosted variants, and the first three substantive claims in answer-section order (answer, foundations, comparison, conclusion), up to 12 claims total. Review additional claims only as a transparently labeled supplement. Retain missing/failed answers in the sample record rather than replacing them with easier successful cases.

A reviewer independent of the generating model should open each cited OpenAlex record and DOI/publisher/accessible full-text source when available, read the exact supplied excerpt in the trajectory, and record reviewer identity/date, source accessed, evidence location, verdict and rationale. Do not claim full-text support when only an abstract was accessible. Do not infer support from citation count, a valid DOI or an identical quotation alone.

For each claim assess:

1. Does each reference identify the actual retrieved publication? Is any source fabricated, misattributed, or outside this trajectory?
2. Does accessible evidence support the proposition, including method, population, intervention, outcome, direction, magnitude, conditions and uncertainty?
3. Is the claim narrower than or equivalent to the evidence, rather than an unsupported generalization or causal inference?
4. For comparative claims, are both approaches supported by appropriate sources, and are cross-study design differences respected?
5. For citation/path claims, does the observed source actually cite the target in that direction? Endpoint co-discovery is insufficient.
6. Are contrary findings and limitations represented faithfully? Are unanswered question components explicit?

Allowed verdicts:

- **SUPPORTED:** accessible source evidence supports the claim within its stated scope.
- **PARTLY_SUPPORTED:** a material part is supported, but scope, comparison or certainty exceeds evidence.
- **UNSUPPORTED:** the cited evidence does not establish the proposition.
- **CONTRADICTED:** the cited source provides evidence inconsistent with the proposition.
- **NOT_ASSESSABLE:** source access or available content is insufficient to judge.
- **UNVERIFIED:** review has not been performed; never count this as correct or incorrect.

Separately rate every frozen question component as ADDRESSED_WITH_SUPPORT / PARTLY_ADDRESSED / MISSING / NOT_ASSESSABLE / UNVERIFIED, with supporting claim IDs. Keep citation traceability, answer presence, component coverage and scientific support as separate outcomes. Report denominators, missing claims, reviewer limitations and disagreements; a second review is desirable where feasible. Do not relabel an automated validator or this coding assistant as independent human adjudication.
