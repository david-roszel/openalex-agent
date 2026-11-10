# Publication security review — November 10, 2026

Repository: `david-roszel/openalex-agent`. Audited release: `f76e842f3f468997a5857414499c3af81c869498`. The historical audit verdict was **PASS WITH WARNINGS**, with no confirmed credentials or private research artifacts identified. The repository was private at review; this document does not authorize publication.

## Historical coverage and findings

The audit checked all four reachable commits across local and advertised remote refs, including the PR head, with no tags present. It also inspected one locally unreachable commit: all 404 stored objects, including 352 unique file blobs and five commit messages, were covered. Connectivity and parent closure were verified; no missing objects, LFS payloads or unexamined binary archives were found.

Gitleaks 8.16.0 (distribution package 8.16.0-1build2) scanned all-ref history and all stored blob/message content, complemented by independent credential-pattern and artifact inspection. Five known warnings were independently resolved: a synthetic archive-test sentinel at `tests/test_submission_system.py:293` and four SHA-256 entries at `versions/revision7/SOURCE_MANIFEST.json:15,74,92,103`, recomputed against committed bytes. An additional root/frozen Meta `PROVENANCE.json:13` warning also matched an actual source checksum. No matched values are reproduced. No authentication was attempted to classify these values.

No private grading credentials, sealed held-out inputs, independent human labels, complete corpus, private archive or full raw hosted conversations were identified. Public development/example inputs and small synthetic/authentic metadata fixtures are intentional and are not independent scientific evaluations.

This is a historical credential/data exposure review, not a penetration test, universal secret detector, dependency-vulnerability certification or proof of scientific accuracy. Unadvertised GitHub server garbage/deleted refs, forks, caches, CI/release artifacts, backups and future commits were not inventoried. Coverage applies to the recorded release; later changes need their own checks.

## Cleanup and remaining boundaries

The follow-up cleanup strengthens sensitive filenames and renamed credential-JSON detection, updates portable publishing instructions, and explains licensing. It leaves application behavior, scientific policies and all frozen R7 source unchanged. No global Gitleaks rule suppression was added. Source hashes for the cleanup differ from the audited release and are recorded in the publication manifest; historical experimental identities remain unchanged.

Git history exposes author/committer names and contact addresses. Ordinary workstation paths remain in historical commits and byte-pinned R7 provenance/source; their referenced private data is not included. These disclosures require owner review, not an unrequested history rewrite. A current synthetic security-test path is anonymized. See [licensing boundaries](LICENSING.md) for the unresolved original-code licensing decision and preserved Meta noncommercial restrictions.

## Safe credential and artifact handling

- Put credentials only in ignored local configuration; `.env.example` must remain blank for secrets. Use separate grading keys and keep private grading archives outside any public release.
- Keep downloaded corpora, private questions, annotations, notebook outputs, hosted transcripts, provider receipts and ledgers local. Redaction is not a guarantee that arbitrary user text contains no sensitive information.
- `.gitignore` protects future ordinary additions only; it neither removes tracked/history content nor prevents forced staging. Filename checks are intentionally conservative and cannot identify every renamed private dataset.
- Before each release run `python scripts/repository_audit.py`, `python scripts/verify_manifest.py`, and, after reviewing/staging, `python scripts/repository_audit.py --staged` plus `git diff --cached --check`. Inspect the exact staged diff manually.
- Scan history separately with `gitleaks detect --source . --log-opts="--all --full-history --no-ext-diff --no-textconv" --redact`. An exit reporting findings requires review; documented checksum/test false positives are not blanket exemptions for future edits. Do not upload unredacted scanner output or raw private logs.
- If a real credential is exposed, restrict disclosure and follow the provider's revocation/rotation process before making further release decisions. This preparation found no credential requiring rotation.

For future commits, consider GitHub's private-email settings and an account-specific verified noreply address. From the repository root, after choosing the exact address, use `git config --local user.email 'YOUR_VERIFIED_GITHUB_NOREPLY_ADDRESS'`; this does not change existing commits. Global Git configuration was not changed.

## Reporting suspected vulnerabilities

Do not post credentials, private traces or human annotations in public issues. If GitHub private vulnerability reporting is enabled, use the repository Security → Report a vulnerability workflow. Its availability has not been asserted. Otherwise open a minimal issue requesting a private reporting channel, without exploit details or sensitive attachments, and wait for a maintainer to provide one. Never assume ordinary issue attachments are private.
