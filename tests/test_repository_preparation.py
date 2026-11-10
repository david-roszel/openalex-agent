"""Publication provenance and portable fixtures; no external services required."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openalex_agent import git_provenance as revision
from openalex_agent.corpus import LocalCorpus
from scripts.repository_audit import PATTERNS, audit, local_only


def test_no_git_is_explicit_unknown(monkeypatch):
    monkeypatch.setattr(revision.shutil, "which", lambda _: None)
    assert revision.git_provenance() == {"commit": None, "dirty": None, "status": "git_unavailable"}


@pytest.mark.parametrize("changed", [False, True])
def test_git_commit_and_dirty_state_are_read_only_and_do_not_capture_remotes(monkeypatch, tmp_path, changed):
    monkeypatch.setattr(revision.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setenv("GIT_DIR", "WRONG REPOSITORY")
    commands = []
    outputs = iter(["true\n", "a" * 40 + "\n", " M README.md\n" if changed else ""])

    def run(command, **kwargs):
        commands.append(command)
        assert "GIT_DIR" not in kwargs["env"] and kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        assert kwargs["timeout"] == 5 and kwargs["capture_output"]
        return SimpleNamespace(returncode=0, stdout=next(outputs))

    monkeypatch.setattr(revision.subprocess, "run", run)
    assert revision.git_provenance(tmp_path) == {"commit": "a" * 40, "dirty": changed, "status": "recorded"}
    assert len(commands) == 3 and all("remote" not in c and "config" not in c and "add" not in c for c in commands)


@pytest.mark.parametrize("case,expected", [("not_repo", "not_a_repository"), ("unborn", "no_commits"), ("timeout", "git_error")])
def test_unavailable_revisions_never_invent_commit_or_leak_error(monkeypatch, case, expected):
    monkeypatch.setattr(revision.shutil, "which", lambda _: "git")
    def run(command, **kwargs):
        if case == "timeout":raise subprocess.TimeoutExpired("PRIVATE", 5)
        if "--is-inside-work-tree" in command:
            return SimpleNamespace(returncode=1 if case == "not_repo" else 0, stdout="true\n")
        return SimpleNamespace(returncode=1, stdout="PRIVATE ERROR")
    monkeypatch.setattr(revision.subprocess, "run", run)
    assert revision.git_provenance() == {"commit": None, "dirty": None, "status": expected}


def test_portable_real_metadata_fixture_is_complete_and_checksummed():
    directory=Path(__file__).parent/"fixtures/development-corpus"
    receipt=json.loads((directory/"fixture-provenance.json").read_text())
    assert receipt["count"]==len(LocalCorpus(directory).papers)==39
    for name, digest in receipt["files"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest


def test_publication_scan_redacts_configured_secrets_and_rejects_staged_local_files(tmp_path,monkeypatch):
    from scripts import repository_audit as scanner
    secret="sk-"+"private"*8
    (tmp_path/".env").write_text("LLM_API_KEY="+secret+"\n")
    (tmp_path/"README.md").write_text("Accidentally included: "+secret)
    report=audit(tmp_path)
    assert secret not in json.dumps(report)
    assert any(c["path"]=="README.md" and "sensitive" in c["type"] for c in report["concerns"])
    monkeypatch.setattr(scanner,"staged_candidates",lambda _:[(".env",False,secret.encode()),("data/results.json",False,b"{}")])
    staged=audit(tmp_path,staged=True)
    assert sum(c["type"]=="local-only artifact included" for c in staged["concerns"])==2
    assert secret not in json.dumps(staged)
    assert PATTERNS["provider-key pattern"].search(secret.encode())


def test_publication_scan_does_not_follow_symlink_to_secret(tmp_path):
    private=tmp_path/"private";private.write_text("PRIVATE MUST NOT BE READ")
    (tmp_path/"README.md").symlink_to(private)
    report=audit(tmp_path)
    assert any(c["type"].startswith("symlink") for c in report["concerns"])
    assert report["inventory"][0]["bytes"]==0


@pytest.mark.parametrize("name",[".env",".env.local","data/frozen/manifest.json",".streamlit/secrets.toml",".venv/a","local.pem","a.sqlite-wal","credentials/grader.env","private-dist/grading.zip","developer.env","questions.private.json","benchmark_heldout.json"])
def test_local_files_excluded_without_excluding_fixtures(name):
    assert local_only(name)
    assert not local_only(".env.example")
    assert not local_only("tests/fixtures/development-corpus/papers.jsonl")
    assert not local_only("openalex_agent/tokenizer_data/o200k_base.tiktoken")
