"""日期格式转换：YYYYMMDD / datetime / DolphinDB YYYY.MM.DD。"""

from __future__ import annotations

import datetime as dt
from typing import Union

DateLike = Union[str, dt.date, dt.datetime]


def to_datetime(value: DateLike) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return dt.datetime.strptime(s[:10].replace("/", "-") if fmt == "%Y-%m-%d" else s, fmt)
        except ValueError:
            continue
    # 兼容 2024.01.01 被截断
    try:
        return dt.datetime.strptime(s.replace("-", ".")[:10], "%Y.%m.%d")
    except ValueError as exc:
        raise ValueError(f"无法解析日期: {value}") from exc


def to_ddb_date(value: DateLike) -> str:
    """Python 日期 -> DolphinDB 字面量 `2024.01.01`（不含引号）。"""
    d = to_datetime(value)
    return d.strftime("%Y.%m.%d")


def to_yyyymmdd(value: DateLike) -> str:
    return to_datetime(value).strftime("%Y%m%d")
