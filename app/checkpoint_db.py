"""Checkpoint schema contract, shared by the isolated job and read-only startup."""
import os
import re

from langgraph.checkpoint.postgres.base import MIGRATIONS
from psycopg import sql

TABLES = ("checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes")
INDEXES = ("checkpoints_thread_id_idx", "checkpoint_blobs_thread_id_idx", "checkpoint_writes_thread_id_idx")


def checkpoint_schema(value: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value):
        raise RuntimeError("LANGGRAPH_SCHEMA must be a lowercase PostgreSQL identifier")
    return value


async def validate_history(conn, schema: str, *, complete: bool) -> None:
    cursor = await conn.execute(sql.SQL("SELECT v FROM {}.checkpoint_migrations ORDER BY v").format(sql.Identifier(schema)))
    versions = [row["v"] for row in await cursor.fetchall()]
    expected = list(range(len(MIGRATIONS) if complete else len(versions)))
    if versions != expected or len(versions) > len(MIGRATIONS):
        raise RuntimeError("checkpoint schema version mismatch")


async def validate_indexes(conn, schema: str) -> None:
    cursor = await conn.execute("""
        SELECT c.relname FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = ANY(%s)
          AND i.indisvalid AND i.indisready
    """, (schema, list(INDEXES)))
    if {row["relname"] for row in await cursor.fetchall()} != set(INDEXES):
        raise RuntimeError("checkpoint indexes require operator recovery")


async def validate_runtime_schema(conn, schema: str) -> None:
    if os.environ.get("AGENT_CHECKPOINT_MIGRATION_DSN"):
        raise RuntimeError("Agent serving must not receive migration credentials")
    checkpoint_schema(schema)
    cursor = await conn.execute("""
        SELECT current_user <> 'map_agent_runtime'
          OR rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls
          OR has_database_privilege(current_user, current_database(), 'CREATE')
          OR has_schema_privilege(current_user, %s, 'CREATE')
          OR has_schema_privilege(current_user, 'public', 'CREATE')
          OR pg_has_role(current_user, 'map_agent_owner', 'MEMBER')
          OR (SELECT nspowner <> 'map_agent_owner'::regrole FROM pg_namespace WHERE nspname=%s) AS elevated
        FROM pg_roles WHERE rolname = current_user
    """, (schema, schema))
    if (await cursor.fetchone())["elevated"]:
        raise RuntimeError("Agent runtime must use a restricted role")
    await validate_history(conn, schema, complete=True)
    cursor = await conn.execute("""
        SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = ANY(%s) AND c.relkind = 'r'
    """, (schema, list(TABLES)))
    if {row["relname"] for row in await cursor.fetchall()} != set(TABLES):
        raise RuntimeError("checkpoint tables missing")
    await validate_indexes(conn, schema)
