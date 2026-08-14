"""Pilot-universe SW L1 industry mapping (hardcoded; no Datayes round-trip)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


# 申万一级（简化名）— 当前 10 票试点
SYMBOL_INDUSTRY: Dict[str, str] = {
    "600519.SH": "白酒",
    "000858.SZ": "白酒",
    "300750.SZ": "电池",
    "601012.SH": "光伏",
    "002594.SZ": "汽车",
    "600036.SH": "银行",
    "601318.SH": "保险",
    "600900.SH": "电力",
    "000333.SZ": "家电",
    "300059.SZ": "证券",
}


def industry_vocab(symbols: Sequence[str] = None) -> List[str]:
    if symbols is None:
        names = sorted(set(SYMBOL_INDUSTRY.values()))
    else:
        names = sorted({SYMBOL_INDUSTRY.get(s, "其他") for s in symbols})
    return names


def industry_ids_for_symbols(symbols: Sequence[str]) -> Tuple[List[int], List[str]]:
    vocab = industry_vocab(symbols)
    name_to_id = {n: i for i, n in enumerate(vocab)}
    ids = [name_to_id[SYMBOL_INDUSTRY.get(s, "其他")] for s in symbols]
    return ids, vocab
