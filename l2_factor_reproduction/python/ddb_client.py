"""DolphinDB 会话封装：连接、执行脚本、上传表。

复用 ``factor_data_loaders.connect_ddb`` / ``core.ddb.connection.get_ddb_session``。
默认共享会话，调用 ``close()`` 不会关闭共享连接（避免误伤同进程其他研究脚本）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.ddb.connection import (
    close_shared_ddb_session,
    get_ddb_session,
    is_shared_session,
)

logger = logging.getLogger(__name__)


class DDBFactorClient:
    def __init__(self, session=None, *, reuse: bool = True):
        self._owns_session = session is None and not reuse
        if session is not None:
            self.session = session
        else:
            # reuse=True 时走共享会话；False 时单独建连，允许 close
            self.session = get_ddb_session(reuse=reuse)
            self._owns_session = not reuse

    def run_script(self, script: str) -> Any:
        """执行 DDB 脚本；表结果通常自动转为 pandas.DataFrame。"""
        return self.session.run(script)

    def run_file(self, abs_path: str) -> Any:
        """加载本地 ``.dos`` 文件内容并 ``session.run(脚本字符串)``。

        说明：``pyread`` 等账号通常 **无权** 执行服务端 ``run("/path.dos")``，
        因此改为在客户端读文件后以脚本字符串提交（与 ``intraday_lib.ddb_functions`` 用法一致）。
        """
        path = abs_path.replace("\\", "/")
        with open(path, "r", encoding="utf-8") as fh:
            script = fh.read()
        if not script.strip():
            raise ValueError(f"empty DDB script: {path}")
        logger.debug("run local DDB script (%d chars): %s", len(script), path)
        return self.session.run(script)

    def upload_table(self, name: str, df: pd.DataFrame) -> None:
        self.session.upload({name: df})

    def close(self, *, force_shared: bool = False) -> None:
        """关闭会话。

        - 独占会话：直接 close
        - 共享会话：默认 no-op；仅 ``force_shared=True`` 时调用 ``close_shared_ddb_session``
        """
        if self._owns_session:
            try:
                self.session.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("close dedicated session failed: %s", exc)
            return
        if force_shared and is_shared_session(self.session):
            close_shared_ddb_session()
