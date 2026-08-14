# Appendix A — Formula details (pointers)

Canonical math lives in Packs; this appendix indexes them.

| Factor | Formula path |
|--------|--------------|
| TGD20 | `research/reports/factors/TGD20/formula.md` |
| D1 | `research/reports/factors/D1_LiquidityQuality60d/formula.md` |
| FlowDensity20 | `research/reports/factors/FlowDensity20/formula.md` |
| APM_SessionResidual | Pack `summary.yaml` `formula:` + Section 4.4 |
| IdealReversal | `research/reports/factors/IdealReversal/formula.md` |
| IdealAmplitude | `research/reports/factors/IdealAmplitude/formula.md` |
| SmartMoney10d | `research/reports/smart_money_v1/` (no Pack formula.md yet) |

# Appendix B — Data schema (high level)

| Layer | Typical store |
|-------|----------------|
| Minute / L2 | DolphinDB `dfs://QV_Trade_to_MinuteBar` etc. |
| EOD | OHLCV + mktcap + industry |
| Feature cache | `research/cache/` (e.g. `apm_session/`, `sue_consensus_eps/`) |
| Pack artifacts | `research/reports/factors/<id>/` |

# Appendix C — Code map (research path)

| Concern | Location |
|---------|----------|
| TGD | `core/l2_features/tgd*.py`, `return_timing.py`, `timing_residual.py` |
| APM session | `core/l2_features/apm_session_*.py`, `run_milestone_c1_apm_session_*.py` |
| D1 / Flow EOD-L2 | `factor_formulas_liquidity_d1.py`, `factor_formulas_l2_flow_p2.py` |
| Ideal cutting | `factor_cutting/ideal_*.py`, `factor_cutting/engine.py` |
| Specs | `factor_specs/*.yaml` |
| Library CSV | `research/reports/factors/factor_library.csv` |

# Appendix D — Provenance rule

If a number in the memo conflicts with Pack `summary.yaml` / CSV, **Pack wins**. Report should be corrected, not the Pack silently overridden.
