# Publish the prepared update after review

This candidate is an uncommitted branch based on the existing public main at9b6d564f1e2f1c162ad487710947aa2c1530ce54, with the original remote/history preserved. No private grading key/archive, corpus, held-out question file, approval, human annotation or raw experiment output is included. Original working checkouts and private submission ZIP are untouched.

Review the current candidate before executing these commands. They create a commit and publish a feature branch; neither action was executed automatically. Do not force-push or reinitialize Git.

```bash
cd /home/bs/Documents/OpenAlex-CLM-GitHub-Update-2026-11-10
git status --short --branch
git remote get-url origin
# Expected: https://github.com/david-roszel/openalex-agent
.venv/bin/python scripts/repository_audit.py
.venv/bin/python scripts/verify_manifest.py
git diff --check
git add -- .gitattributes .gitignore .env.example .github .streamlit/config.toml README.md PUBLISHING.md PUBLIC_MANIFEST.json pyproject.toml requirements.txt requirements.lock.txt requirements-ui.lock.txt requirements-gemini.txt main.py app.py live_app.py legacy_app.py launch-dashboard.sh launch-live-research.sh openalex_agent tests benchmarks docs scripts versions integration examples tools
.venv/bin/python scripts/repository_audit.py --staged
git diff --cached --check
git diff --cached --stat
git diff --cached
# Only after approving the exact staged diff:
git commit -m "Publish live research, five-method evaluation UI and frozen R7 reference"
git fetch origin
# Inspect any new remote changes before integration; do not force-push.
git push --dry-run origin HEAD:refs/heads/prepare/public-refresh-2026-11-10
git push -u origin prepare/public-refresh-2026-11-10
```

Then, if GitHub CLI is installed/authenticated, open the review PR:

```bash
gh pr create --base main --head prepare/public-refresh-2026-11-10 \
  --title "Publish live research and scientific evaluation workflows" \
  --body-file docs/RELEASE_NOTES.md
```

Keep the resulting PR unmerged until its CI and review are complete. No commit, push, PR, merge or visibility change was performed by preparation. The public checkout main is left unchanged; this separate worktree shares history/remote.

For later intentional edits, regenerate `PUBLIC_MANIFEST.json` with `python scripts/update_public_manifest.py`, then repeat audit/test/diff review. A manifest is provenance, not approval or proof of scientific validity. The working-tree scan is the preparation check; exact index scanning applies only after the user's staging command. Do not treat an old clean index audit as a scan of unstaged new files.

The scan covers allowlisted current publication paths, explicit private filenames, configured-secret patterns and blank templates. It does not certify every unknown credential format or old Git history. Meta/tokenizer attribution is retained; a blanket project license remains an owner decision. Never use the private grading ZIP as the source for this push.
