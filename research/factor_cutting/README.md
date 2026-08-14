# Factor Cutting Research Platform

**v1** = prove cutting works  
**v2** = replication + mechanism + knife ranking  
**v2.1** = neutralization ladder + knife families + `search_knives` API  

Minute APM / Smart Money: wait until daily knife space + neut alpha are stable.

## v2.1 smoke headline (2023–2025)

Neutralization (`ideal_reversal` / ATS):

| Mode | RankIC | Retention |
|------|--------|-----------|
| raw | −3.77% | 1.00 |
| size | −3.74% | **0.99** |
| size+industry | −3.31% | **0.88** |

→ Survives size/industry → not a pure size artifact.

Knife families: `volume`/`amount` (participation) top score but corr≈0.92 (same story).  
`ats_trade_count` independent of `trade_count` (corr 0.17).  
`search_knives` → best=`volume`, independent=`[ats_trade_count, amihud]`.

## Commands

```bash
# v2.1 research platform
OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v21.py --preset smoke
OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v21.py --preset ddb

# Paper 2010–2025 (Oracle, long)
OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v2.py --preset paper --keep-cache
```

API:

```python
from factor_cutting.info_layer import search_knives
result = search_knives(ret_1d, fwd_ret, amount=amount, volume=volume, trade_count=tc)
# result["best_knife"], result["independent_knives"]
```
