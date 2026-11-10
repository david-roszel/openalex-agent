"""Credential-safe diagnostics and durable per-request conservative accounting."""
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import traceback

from .openalex import BudgetExceeded


def scrub(value, secrets=()):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"https?://[^\s\"']+", "[URL REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    return value


def diagnostic(exc, secrets=()):
    return {"type": type(exc).__name__, "message": scrub(str(exc), secrets),
        "traceback": scrub("".join(traceback.format_exception(exc)), secrets)}


class SpendingLedger:
    """Independent LiveDemo ledger; never imports or opens V3 accounting.

    Hold before HTTP transmission; settle conservative token-price bounds.
    Missing usage or interrupted attempts keep their hold and block later calls.
    A settled/uncertain request key can never be transmitted again.
    """
    def __init__(self, path, *, ceiling, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ceiling = Decimal(str(ceiling))
        if not self.ceiling.is_finite() or self.ceiling <= 0:
            raise ValueError("Spending ceiling must be a finite positive amount")
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS configuration(identity TEXT, ceiling TEXT)")
            row = db.execute("SELECT identity,ceiling FROM configuration").fetchone()
            if row and (row[0] != identity or Decimal(row[1]) != self.ceiling):
                raise ValueError("Ledger identity or authorized ceiling changed; use its original configuration")
            if not row:
                db.execute("INSERT INTO configuration VALUES(?,?)", (identity, str(self.ceiling)))
            db.execute("CREATE TABLE IF NOT EXISTS attempts(key TEXT PRIMARY KEY, trajectory TEXT, status TEXT, held TEXT, charge TEXT, request_sha256 TEXT, created REAL, usage TEXT)")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, key, trajectory, wire, *, input_price, output_price, max_output):
        # UTF-8 bytes bound token count; add framing allowance. No token cache
        # discount or credits. Short context, pinned Standard tier only.
        reserve = (Decimal(len(wire) + 1024) * Decimal(str(input_price)) +
                   Decimal(max_output) * Decimal(str(output_price))) / Decimal(1000000)
        with self.transaction() as db:
            rows = db.execute("SELECT status,held FROM attempts").fetchall()
            if any(status != "SETTLED" for status, _ in rows):
                raise BudgetExceeded("Unsettled or unknown paid request; reconcile before continuing")
            if db.execute("SELECT 1 FROM attempts WHERE key=?", (key,)).fetchone():
                raise BudgetExceeded("Request already attempted; automatic replay is forbidden")
            if sum((Decimal(held) for _, held in rows), Decimal(0)) + reserve > self.ceiling:
                raise BudgetExceeded("Independent LiveDemo spending ceiling reached")
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)", (key, trajectory, "RESERVED",
                str(reserve), None, hashlib.sha256(wire).hexdigest(), time.time(), None))
        return key

    def ensure_affordable(self, wire, *, input_price, output_price, max_output):
        reserve = (Decimal(len(wire) + 1024) * Decimal(str(input_price)) +
                   Decimal(max_output) * Decimal(str(output_price))) / Decimal(1000000)
        summary = self.summary()
        if summary["uncertain_attempts"]:
            raise BudgetExceeded("Unsettled or unknown paid request; reconcile before continuing")
        if Decimal(summary["retained_upper_bound_usd"]) + reserve > self.ceiling:
            raise BudgetExceeded("Independent LiveDemo spending ceiling reached")

    def settle(self, key, usage, *, input_price, output_price, max_output, service_tier=None):
        incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        valid = (type(incoming) is int and incoming >= 0 and type(outgoing) is int and
                 0 <= outgoing <= max_output and service_tier in (None, "default", "standard"))
        charge = ((Decimal(incoming) * Decimal(str(input_price)) + Decimal(outgoing) * Decimal(str(output_price))) /
                  Decimal(1000000)) if valid else None
        with self.transaction() as db:
            row = db.execute("SELECT status,held FROM attempts WHERE key=?", (key,)).fetchone()
            if not row or row[0] != "RESERVED":
                raise BudgetExceeded("Cannot reconcile a non-pending request")
            if not valid or charge > Decimal(row[1]):
                db.execute("UPDATE attempts SET status='UNKNOWN',usage=? WHERE key=?", (json.dumps(usage), key))
            else:
                db.execute("UPDATE attempts SET status='SETTLED',held=?,charge=?,usage=? WHERE key=?",
                    (str(charge), str(charge), json.dumps(usage), key))
        if not valid or charge > Decimal(row[1]):
            raise BudgetExceeded("Unknown or inconsistent provider usage; spending hold retained")

    def summary(self):
        with self.transaction() as db:
            rows = db.execute("SELECT status,held,charge FROM attempts").fetchall()
        return {"ceiling_usd": str(self.ceiling), "attempts": len(rows),
            "retained_upper_bound_usd": str(sum((Decimal(r[1]) for r in rows), Decimal(0))),
            "usage_based_upper_bound_usd": str(sum((Decimal(r[2]) for r in rows if r[2] is not None), Decimal(0))),
            "uncertain_attempts": sum(r[0] != "SETTLED" for r in rows),
            "basis": "Provider-reported tokens at conservative configured prices; not an invoice or assumed discount"}
