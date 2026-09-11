"""Run the pinned checkpoint migrations once, without serving/Redis/Gemini imports."""
import asyncio
import os
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row

from app.checkpoint_db import checkpoint_schema, validate_history, validate_indexes

LOCK_ID = 684821308


async def migrate(dsn: str, schema: str) -> None:
    schema = checkpoint_schema(schema)
    # CREATE INDEX CONCURRENTLY in the pinned migrations requires autocommit.
    # A session lock stays held on this same connection across every migration.
    async with await AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row,
                                            connect_timeout=10) as conn:
        cursor = await conn.execute("""
            SELECT rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls
            AS elevated FROM pg_roles WHERE rolname = session_user
        """)
        if (await cursor.fetchone())["elevated"]:
            raise RuntimeError("Agent migrator must be a restricted login")
        await conn.execute("SET ROLE map_agent_owner")
        await conn.execute("SET lock_timeout = '10s'")
        await conn.execute("SET statement_timeout = '5min'")
        cursor = await conn.execute("SELECT pg_try_advisory_lock(%s) AS locked", (LOCK_ID,))
        if not (await cursor.fetchone())["locked"]:
            raise RuntimeError("Agent migration already running")
        try:
            cursor = await conn.execute("SELECT to_regnamespace(%s) AS schema", (schema,))
            if (await cursor.fetchone())["schema"] is None:
                raise RuntimeError("Agent schema must be provisioned before migration")
            await conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            cursor = await conn.execute("SELECT to_regclass(%s) AS history", (schema + ".checkpoint_migrations",))
            if (await cursor.fetchone())["history"] is not None:
                await validate_history(conn, schema, complete=False)
            await AsyncPostgresSaver(conn).setup()
            await validate_history(conn, schema, complete=True)
            await validate_indexes(conn, schema)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))


def main() -> int:
    dsn = os.environ.get("AGENT_CHECKPOINT_MIGRATION_DSN", "")
    if not dsn or os.environ.get("POSTGRES_PASSWORD"):
        print("Agent migration requires its dedicated DSN and no runtime password", file=sys.stderr)
        return 1
    try:
        asyncio.run(migrate(dsn, os.environ.get("LANGGRAPH_SCHEMA", "langgraph")))
    except Exception as error:
        import re
        state = getattr(error, "sqlstate", None)
        safe_state = state if isinstance(state, str) and re.fullmatch(r"[A-Z0-9]{5}", state) else "unknown"
        print(f"Agent checkpoint migration failed; error_class={type(error).__name__}; sqlstate={safe_state}", file=sys.stderr)
        return 1
    print("Agent checkpoint migration completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
