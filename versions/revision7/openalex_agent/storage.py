"""SQLite metadata, durable request cache, resumable jobs, and audit events."""

import json
from pathlib import Path
import sqlite3
from typing import Iterable
from .models import Paper, canonical_id, utcnow


class Database:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS papers(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, payload TEXT NOT NULL, acquired TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run_id TEXT, kind TEXT, timestamp TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
        except sqlite3.Error:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def save_papers(self, papers: Iterable[Paper]) -> None:
        """Upsert by ID without letting a historical request cache roll metadata back."""
        with self.connection:
            self.connection.executemany("""INSERT INTO papers VALUES (?,?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
                WHERE json_extract(excluded.payload, '$.retrieved_at') >= json_extract(papers.payload, '$.retrieved_at')""", (
                (p.id, json.dumps(p.to_dict(), ensure_ascii=False)) for p in papers))

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.connection.execute("SELECT payload FROM papers WHERE id=?", (canonical_id(paper_id),)).fetchone()
        return Paper.from_dict(json.loads(row[0])) if row else None

    def papers(self) -> list[Paper]:
        return [Paper.from_dict(json.loads(row[0])) for row in self.connection.execute("SELECT payload FROM papers ORDER BY id")]

    def cache_get(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_acquired(self, key: str) -> str:
        return self.connection.execute("SELECT acquired FROM cache WHERE key=?", (key,)).fetchone()[0]

    def cache_put(self, key: str, payload: dict) -> str:
        acquired = utcnow()
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(payload), acquired))
        return acquired

    def event(self, run_id: str, kind: str, payload: dict) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO events(run_id,kind,timestamp,payload) VALUES (?,?,?,?)", (
                run_id, kind, utcnow(), json.dumps(payload, ensure_ascii=False)))

    def events(self, run_id: str) -> list[dict]:
        return [{"kind": k, "timestamp": t, **json.loads(p)} for k, t, p in self.connection.execute(
            "SELECT kind,timestamp,payload FROM events WHERE run_id=? ORDER BY seq", (run_id,))]

    def job_get(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM jobs WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def job_put(self, key: str, payload: dict) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (key, json.dumps(payload)))
