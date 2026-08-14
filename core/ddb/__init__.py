"""DolphinDB access — single entry for sessions and table reads."""

from core.ddb.connection import get_ddb_session

__all__ = ["get_ddb_session"]
