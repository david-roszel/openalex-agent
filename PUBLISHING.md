# Initialize and publish this export later

This directory was exported without `.git`, commits, remotes, credentials or
private experiments. Nothing was initialized, committed, pushed or made public
automatically. Inspect the release, choose a code license and author/citation
metadata, and approve publication before performing any remote action.

From this directory, in a shell where Git and the documented Python environment
are available, confirm there is no inherited parent repository:

```bash
pwd
git rev-parse --show-toplevel
```

The second command should report no repository before initialization. If it finds
an existing repository, stop and inspect rather than replacing it. Then:

```bash
git init -b main
python scripts/repository_audit.py
python -m pytest
git add -- .gitignore .env.example .streamlit .github README.md PUBLISHING.md PUBLIC_MANIFEST.json pyproject.toml requirements.txt requirements.lock.txt requirements-ui.lock.txt main.py app.py launch-dashboard.sh openalex_agent tests benchmarks docs scripts
python scripts/repository_audit.py --staged
git diff --cached --check
git diff --cached --stat
git status --short
git commit -m "feat: release OpenAlex CLM research implementation"
```

Do not add the separate private export-review directory or any private source
repository history. No commit was created by preparing these instructions.
Configure author identity in your own Git installation rather than inventing it.

Only after explicitly approving publication, create an empty GitHub repository
in your own account with your chosen visibility. Replace the placeholders below
with that repository. Use your credential manager/SSH, never a token in a URL.
If a remote already exists, inspect it; do not overwrite it.

```bash
git remote -v
git remote add origin https://github.com/YOUR_OWNER/YOUR_PUBLIC_REPOSITORY.git
git remote get-url origin
git push --dry-run origin main
git push -u origin main
```

These commands use no force. If the destination has existing history or branch
protection rejects the push, stop and inspect. Visibility changes and publication
are owner decisions; the presence of these commands is not approval to execute them.
