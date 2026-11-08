# Publish the reviewed update

This is an update to the existing public repository. Preserve its Git history and
`origin`; do not reinitialize Git or add a replacement remote. The prepared update
contains implementation, synthetic tests, public technical documentation and
vendored attribution. It excludes private questions, annotations, credentials,
frozen corpus, historical traces and generated experiment outputs.

After reviewing this working tree and explicitly approving publication:

```bash
cd /home/bs/Documents/OpenAlex-CLM-Public
git status --short
git remote get-url origin
# Expected: https://github.com/david-roszel/openalex-agent
python scripts/repository_audit.py
python scripts/verify_manifest.py
python -m pytest
git add -- .gitignore .env.example .github .streamlit/config.toml README.md PUBLISHING.md PUBLIC_MANIFEST.json pyproject.toml requirements.txt requirements.lock.txt requirements-ui.lock.txt requirements-gemini.txt main.py app.py launch-dashboard.sh openalex_agent tests benchmarks docs scripts
python scripts/repository_audit.py --staged
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Update seven-method agents and CLM capacity termination policy"
git push --dry-run origin HEAD:main
git push origin HEAD:main
```

Use an environment containing the documented development/UI dependencies. The
optional isolated Gemini SDK is not required for mocked provider tests; SDK-specific
tests may skip without it. Do not force-push or change visibility. If remote
history diverged, stop and inspect before choosing an integration strategy.
The scan is conservative, not certification of unknown secret formats or licenses.
Meta components retain their separate CC BY-NC 4.0 license and NOTICE; the
project-wide license and citation metadata remain owner decisions.
