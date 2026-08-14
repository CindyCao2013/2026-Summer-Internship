"""Unified DolphinDB session management.

All research code should obtain connections via ``get_ddb_session()``.
Legacy ``factor_data_loaders.connect_ddb()`` delegates here for compatibility.

By default ``reuse=True`` returns a process-wide shared session (lazy init).
Callers must not ``close()`` a shared session; use ``close_shared_ddb_session()``
at process shutdown if needed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import dolphindb as ddb

from COMMON_CONST import DATA_DB_CONN

_DEFAULT_CONN: Dict[str, Any] = dict(DATA_DB_CONN)
_shared_session: Optional[ddb.session] = None
_shared_conn_key: Optional[tuple] = None


def _conn_key(conn: dict) -> tuple:
    return tuple(sorted(conn.items()))


def get_ddb_session(
    conn: Optional[dict] = None,
    *,
    reuse: bool = True,
) -> ddb.session:
    """Return a DolphinDB session (shared by default).

    Parameters
    ----------
    conn:
        Connection dict. When omitted, uses ``DATA_DB_CONN``.
    reuse:
        If True and ``conn`` is default, reuse lazy-initialized shared session.
        If False, always open a dedicated session (caller may close it).
    """
    global _shared_session, _shared_conn_key

    cfg = conn or _DEFAULT_CONN
    key = _conn_key(cfg)

    if reuse and conn is None:
        if _shared_session is None or _shared_conn_key != key:
            if _shared_session is not None:
                try:
                    _shared_session.close()
                except Exception:  # noqa: BLE001
                    pass
            _shared_session = ddb.session()
            _shared_session.connect(**cfg)
            _shared_conn_key = key
        return _shared_session

    s = ddb.session()
    s.connect(**cfg)
    return s


def is_shared_session(session) -> bool:
    """True if ``session`` is the process-wide shared session."""
    return _shared_session is not None and session is _shared_session


def close_shared_ddb_session() -> None:
    """Close and clear the shared session (e.g. at job end)."""
    global _shared_session, _shared_conn_key
    if _shared_session is not None:
        try:
            _shared_session.close()
        except Exception:  # noqa: BLE001
            pass
    _shared_session = None
    _shared_conn_key = None
