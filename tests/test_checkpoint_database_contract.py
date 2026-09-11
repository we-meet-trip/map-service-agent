import asyncio
import os
import subprocess
import sys

import pytest

from app.checkpoint_db import MIGRATIONS, checkpoint_schema, validate_history, validate_runtime_schema
from app.checkpoint_migrate import main


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows

    async def fetchone(self):
        return self.rows[0]


class Connection:
    def __init__(self, versions, elevated=False):
        self.versions, self.elevated = versions, elevated
        self.statements = []

    async def execute(self, statement, params=None):
        statement = str(statement)
        self.statements.append(statement)
        if "FROM pg_roles" in statement:
            return Cursor([{"elevated": self.elevated}])
        return Cursor([{"v": v} for v in self.versions])


@pytest.mark.parametrize("value", ["public; DROP SCHEMA public", "../other", "Capital", "한글", "x" * 64, ""])
def test_schema_identifier_rejected(value):
    with pytest.raises(RuntimeError):
        checkpoint_schema(value)


@pytest.mark.parametrize("versions", [[], [0, 2], list(range(len(MIGRATIONS)-1)), list(range(len(MIGRATIONS)+1))])
def test_complete_history_rejects_gaps_old_future(versions):
    with pytest.raises(RuntimeError, match="version mismatch"):
        asyncio.run(validate_history(Connection(versions), "langgraph", complete=True))


def test_pinned_complete_history_and_partial_prefix():
    asyncio.run(validate_history(Connection(list(range(len(MIGRATIONS)))), "langgraph", complete=True))
    asyncio.run(validate_history(Connection([0, 1]), "langgraph", complete=False))


def test_elevated_runtime_rejected_before_history(monkeypatch):
    monkeypatch.delenv("AGENT_CHECKPOINT_MIGRATION_DSN", raising=False)
    conn = Connection([], elevated=True)
    with pytest.raises(RuntimeError, match="restricted role"):
        asyncio.run(validate_runtime_schema(conn, "langgraph"))
    assert len(conn.statements) == 1


def test_serving_rejects_migration_secret_before_db(monkeypatch):
    monkeypatch.setenv("AGENT_CHECKPOINT_MIGRATION_DSN", "private")
    conn = Connection([])
    with pytest.raises(RuntimeError, match="must not receive"):
        asyncio.run(validate_runtime_schema(conn, "langgraph"))
    assert conn.statements == []


def test_job_does_not_import_provider_or_serving_modules():
    code = "import app.checkpoint_migrate, sys; assert not any(x in sys.modules for x in ['app.main', 'app.agent_settings', 'app.clients.agent_clients'])"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))


def test_job_rejects_runtime_password(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_CHECKPOINT_MIGRATION_DSN", "private-dsn")
    monkeypatch.setenv("POSTGRES_PASSWORD", "private-runtime")
    assert main() == 1
    assert "private-" not in capsys.readouterr().err
