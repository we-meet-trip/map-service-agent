from types import SimpleNamespace

import pytest
from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr

from app.main import _checkpoint_conninfo


@pytest.mark.parametrize("password", ["synthetic@p/a:ss#word", "a b'c\\d%25", ""])
def test_postgres_credentials_survive_libpq_parsing(password):
    settings = SimpleNamespace(
        POSTGRES_USER="checkpoint", POSTGRES_PASSWORD=SecretStr(password),
        POSTGRES_HOST="127.0.0.1", POSTGRES_PORT=5432, POSTGRES_DB="map_db",
    )
    parsed = conninfo_to_dict(_checkpoint_conninfo(settings))
    assert parsed == {
        "user": "checkpoint", "password": password, "host": "127.0.0.1",
        "port": "5432", "dbname": "map_db",
    }
