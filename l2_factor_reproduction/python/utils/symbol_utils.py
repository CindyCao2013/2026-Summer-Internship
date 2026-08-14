"""代码格式转换：纯数字 / 交易所后缀互转。"""

from __future__ import annotations


def to_wind_symbol(code: str) -> str:
    """600000 / 600000.SH / sh600000 -> 600000.SH"""
    s = str(code).strip().upper()
    if s.endswith((".SH", ".SZ", ".BJ")):
        return s
    if s.startswith(("SH", "SZ", "BJ")) and len(s) >= 8:
        return f"{s[2:]}.{s[:2]}"
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"无法识别股票代码: {code}")
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("4", "8")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def to_pure_code(symbol: str) -> str:
    """600000.SH -> 600000"""
    return str(symbol).split(".")[0]
