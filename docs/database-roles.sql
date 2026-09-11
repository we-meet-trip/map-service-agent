\set ON_ERROR_STOP on
-- Run as a dedicated provisioning operator against the explicitly selected DB.
-- No passwords; R7 sets LOGIN credentials through its secret channel separately.
-- Reapply after every migration. Existing rows/history and other roles' ACLs survive.
BEGIN;
DO $roles$
DECLARE role_name text;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['map_agent_owner','map_agent_migrator','map_agent_runtime'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
      EXECUTE format('CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT', role_name);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)) THEN
      RAISE EXCEPTION 'Refusing elevated existing service role';
    END IF;
  END LOOP;
END
$roles$;
ALTER ROLE map_agent_owner NOLOGIN NOINHERIT;
ALTER ROLE map_agent_migrator LOGIN NOINHERIT;
ALTER ROLE map_agent_runtime LOGIN NOINHERIT;
GRANT map_agent_owner TO map_agent_migrator;
REVOKE map_agent_owner FROM map_agent_runtime;
SELECT format('GRANT CONNECT ON DATABASE %I TO map_agent_runtime, map_agent_migrator', current_database()) \gexec
CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION map_agent_owner;
ALTER SCHEMA langgraph OWNER TO map_agent_owner;
REVOKE CREATE ON SCHEMA langgraph FROM PUBLIC;
REVOKE ALL ON SCHEMA langgraph FROM map_agent_runtime;
GRANT USAGE ON SCHEMA langgraph TO map_agent_runtime;
DO $ownership$
DECLARE obj record;
BEGIN
  FOR obj IN SELECT c.relname, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE NOT (c.relkind='S' AND EXISTS (SELECT 1 FROM pg_depend d WHERE d.objid=c.oid AND d.deptype IN ('a','i')))
      AND n.nspname='langgraph' AND c.relkind IN ('r','p','S','v','m')
  LOOP
    EXECUTE format('ALTER %s langgraph.%I OWNER TO map_agent_owner',
      CASE obj.relkind WHEN 'S' THEN 'SEQUENCE' WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW' ELSE 'TABLE' END, obj.relname);
  END LOOP;
END
$ownership$;
REVOKE ALL ON ALL TABLES IN SCHEMA langgraph FROM map_agent_runtime;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA langgraph FROM map_agent_runtime;
-- New objects stay fail-closed until this explicitly reviewed ACL list is updated.
ALTER DEFAULT PRIVILEGES FOR ROLE map_agent_owner IN SCHEMA langgraph REVOKE ALL ON TABLES FROM map_agent_runtime;
ALTER DEFAULT PRIVILEGES FOR ROLE map_agent_owner IN SCHEMA langgraph REVOKE ALL ON SEQUENCES FROM map_agent_runtime;
DO $acl$
BEGIN
  IF to_regclass('langgraph.checkpoint_migrations') IS NOT NULL THEN
    GRANT SELECT ON langgraph.checkpoint_migrations TO map_agent_runtime;
  END IF;
  IF to_regclass('langgraph.checkpoints') IS NOT NULL THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph.checkpoints TO map_agent_runtime;
  END IF;
  IF to_regclass('langgraph.checkpoint_blobs') IS NOT NULL THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph.checkpoint_blobs TO map_agent_runtime;
  END IF;
  IF to_regclass('langgraph.checkpoint_writes') IS NOT NULL THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON langgraph.checkpoint_writes TO map_agent_runtime;
  END IF;
END
$acl$;
COMMIT;
