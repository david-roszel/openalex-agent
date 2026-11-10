# Review and publish the security cleanup

The recorded release is `f76e842f3f468997a5857414499c3af81c869498` on `origin/main`. It already contains the live application, five-method evaluation UI and isolated R7 reference. The repository is **private** at the November 10, 2026 review; the historical label "public release" describes intended distribution, not a completed visibility change.

This cleanup starts from that release on `prepare/public-security-cleanup-2026-11-10`. It changes publication guards, tests and documentation, not research algorithms or frozen R7 files. It remains uncommitted until the owner reviews it. Earlier release preparation was committed and merged; do not repeat its initialization or staging recipe.

From the repository root, review and validate:

```bash
git status --short --branch
git remote get-url origin
# Expected: https://github.com/david-roszel/openalex-agent
git fetch origin
git log --oneline --decorate -5
# Inspect remote changes before proceeding; never reset or force-push.
.venv/bin/python scripts/repository_audit.py
.venv/bin/python scripts/verify_manifest.py
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=integration/offline_guard:. \
  .venv/bin/python -m pytest -p no:cacheprovider
./scripts/test-revision7-offline.sh
```

After approving the exact changes, stage only the cleanup files and inspect the actual index:

```bash
git add -- .gitignore README.md PUBLISHING.md PUBLIC_MANIFEST.json \
  docs/RELEASE_NOTES.md docs/VALIDATION.md docs/ITERATIONS.md \
  docs/SECURITY_REVIEW.md docs/LICENSING.md \
  scripts/repository_audit.py scripts/update_public_manifest.py \
  tests/test_publication_security.py tests/test_meta_clm.py
.venv/bin/python scripts/repository_audit.py --staged
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Only after approving that staged diff:

```bash
git commit -m "Harden publication safeguards and document security and licensing"
git fetch origin
# Inspect any new commits on origin/main before integration.
git push --dry-run origin HEAD:refs/heads/prepare/public-security-cleanup-2026-11-10
git push -u origin prepare/public-security-cleanup-2026-11-10
gh pr create --base main --head prepare/public-security-cleanup-2026-11-10 \
  --title "Harden public-release safeguards and documentation" \
  --body-file docs/RELEASE_NOTES.md
```

These are owner-operated commands, not actions executed by preparation. Keep the PR unmerged until review and CI complete. Do not reinitialize Git, rewrite attribution, force-push, or change visibility as part of this procedure.

For later edits, regenerate `PUBLIC_MANIFEST.json` with `.venv/bin/python scripts/update_public_manifest.py`, then repeat checks. `published_commit: null` refers to the uncommitted cleanup, not the already committed base release. Manifests cannot hash their own eventual commit and are not scientific approval.

The working-tree audit scans allowlisted candidates. `--staged` scans exact index blobs, including files excluded by `.gitignore`; it must be rerun after final staging. Neither is a complete historical scanner. See [security review](docs/SECURITY_REVIEW.md) for Gitleaks coverage and limitations, and [licensing](docs/LICENSING.md) for owner decisions. Never stage a private grading ZIP, local research outputs, credentials or independent annotations. Do not copy local notebook outputs into release documentation.

Before changing visibility, explicitly review historical contact/workstation metadata and the licensing status. Public visibility does not imply unrestricted reuse. No visibility-changing command is included here.
