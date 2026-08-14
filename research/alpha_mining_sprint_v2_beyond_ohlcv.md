# Alpha Mining Sprint v2 — Beyond OHLCV Map

**Status:** P0 SEALED (useful fail) → P1/P2 execution  
**Date:** 2026-07-10  
**Premise:** OHLCV Base3 (D1/D4/D5) is frozen. New factors must be **orthogonal to Base3**, static, explainable, turnover-controlled.  
**Gate (reuse existing density pipeline):** residual IC vs D1/D4/D5 significant + stack ΔSharpe/ICIR ≥ 0 + single-factor turnover fence.

**P0 sealed:** See `research/reports/sue_density_v1/README.md`. Do not retune SUE.  
**P0 pipeline ready — P1 data source plug-and-play.** Equity incentive Wind table incomplete (~444 rows) → P1 uses major-holder + insider trades first.

---

## 0. What we already know (map legend)

| Zone | Status | Implication |
|------|--------|-------------|
| OHLCV density (D1–D5) | Explored & frozen | **Stop mining day-bar OHLCV variants** |
| D6 value / D7 quality composites | Tested; inactive on Base3 | Rebuild via **industry-neutral + itemized quality**, not new composites of the same fields |
| L2 trade-flow (`cn_*_shock`) | Overlay only; `cn_cancel_shock` retained | Deeper L2 = new formulas on **existing minute bar**, not new DB |
| Events / SUE / northbound / ESG | **Not implemented** | True greenfield |

Mentor filter for any new candidate:

1. Independent of D1/D4/D5 (residual IC)  
2. Static, all-market, multi-year stable  
3. Clear economics (reportable)  
4. Single-factor annual turnover ≲ 100%

---

## 1. Priority queue (2–3 weeks)

| Pri | Direction | Why now | Data readiness (this repo) |
|-----|-----------|---------|----------------------------|
| ~~**P0**~~ | ~~SUE~~ | **SEALED useful fail** | `research/reports/sue_density_v1/` |
| **P1** | 大股东净增持 | resid_t=3.63 vs Base3; stack λ*=0 → satellite/enhancer research | `event_density_v1` — incentive table blocked |
| **P2** | L2 net active flow / mktcap 20d | **ICIR 5.04, resid_t_B3 4.68, resid_t_VOI 7.40, stack +1.02** | `l2_flow_density_v1` — top candidate |
| **P3** | News/ESG / northbound | Pipeline depth | Connections only |

**Rule:** 3–5 clean formulas per direction. No 100-variant grid. Pass density gate or drop.

---

## 2. P0 — SUE / consensus surprise (start here)

> **P0 前置硬要求（执行前锁定）**
> 1. 所有 SUE 因子必须基于「最早已知公告日」构建，包含预告、快报、正式报告。  
> 2. 必须测试**事件持有法**与**日频衰减法**两种信号生成方式。  
> 3. 必须正交化**市值 + 行业**后再看 residual IC（再对 Base3）。  
> 4. 必须包含 `unexpected_profit_notice_surprise_20d`。

### 2.1 Economic story (one paragraph)

Actual earnings vs expected (seasonal RW or analyst consensus). Positive surprise → underreaction / information diffusion → subsequent drift. Information lives on **announcement dates**, not in daily OHLCV paths → residual vs Base3 should be the default hypothesis.

### 2.2 Wind tables already confirmed (2026-07-10 probe)

| Table | Role |
|-------|------|
| `WIND.ASHAREINCOME` | Actuals: `ANN_DT` / `ACTUAL_ANN_DT`, `REPORT_PERIOD`, `NET_PROFIT_*`, `S_FA_EPS_*` |
| `WIND.ASHARECONSENSUSDATA` | Consensus: `EST_DT`, `EPS_AVG` / `NET_PROFIT_AVG`, upgrades/downgrades |
| `WIND.ASHARECONSENSUSROLLINGDATA` | Rolling consensus snapshot |
| `WIND.ASHAREEARNINGEST` | Analyst-level estimates (for revision breadth) |
| `WIND.ASHAREPROFITNOTICE` / `ASHAREPROFITEXPRESS` | Pre-announcement / express — early signal, careful with look-ahead |

### 2.3 First 4 formulas only

| ID | Formula (sketch) | Freq |
|----|------------------|------|
| `sue_np_yoy_z` | \((NP_t - NP_{t-4}) / \sigma(ΔNP_{t-8:t-1})\) point-in-time on `ANN_DT` | Event → daily hold |
| `sue_eps_consensus` | \((EPS_{actual} - EPS_{consensus, pre-ann}) / |EPS_{consensus}|\) | Event |
| `analyst_np_revision_20d` | 20d change in consensus NP (pre-ann window) | Daily slow |
| `profit_notice_mid_surprise` | Midpoint of notice range vs last-year NP (signed) | Event |

Align with existing `factor_finance.normalize_finance_long` / ann-date panel pattern (same as ROE quality).

### 2.4 Validation harness (do not invent new gates)

Reuse pattern of `run_fundamental_validation.py` / density stack:

1. Build daily panels (forward-fill from ann date; no peeking past `ANN_DT`).  
2. Rank IC / ICIR / mono / turnover on ALL + CSI300/500/1000.  
3. Residual IC vs Base3 `{D1,D4,D5}`.  
4. Stack: `Base3 + λ·z(new)`, λ ∈ {0.1, 0.2, 0.3}; require ICIR or Sharpe uplift **and** turnover fence.  
5. Verdict: `new_base` / `enhancer` / `satellite` / `drop`.

### 2.5 Implementation checklist

- [x] `sue_data.py`: Wind Oracle notice / express / income / consensus + earliest-known timeline  
- [x] `factor_formulas_sue.py`: 5 factors incl. `unexpected_profit_notice_surprise_20d`; hold + decay  
- [x] `run_sue_density_v1.py`: size+industry residual + stack vs Base3  
- [x] Artifacts under `research/reports/sue_density_v1/`  

### 2.6 P0 result (504d, 2023-12 → 2025-12)

| Factor | Best mode | ICIR | resid_t vs Base3 | Stack uplift | Verdict |
|--------|-----------|------|------------------|--------------|---------|
| `sue_eps_consensus` | decay | **3.71** | 1.76 | 0 | raw_signal_only (near miss) |
| `unexpected_profit_notice_surprise_20d` | decay | 1.08 | ~0 | 0 | drop / not independent |
| `sue_np_yoy_z` | hold | 0.54 | 0.32 | 0 | raw_signal_only |
| others | — | weak | <2 | 0 | drop |

**Outcome:** Useful fail — pipeline works; no factor clears residual-t≥2 **and** stack uplift vs Base3. Next: P1 events or industry-neutral value rebuild; optional longer confirmation window for `sue_eps_consensus` only.

---

## 3. P1 / P2 / P3 — one-liners until P0 lands

**P1 Events:** Probe Wind for 股权激励 / 回购 / 增减持 tables → event dummies with 60–120d holding → same density gate.  
**P2 L2 flow:** `sum_20d(active_buy_amount - active_sell_amount)` / float mktcap from existing minute bar; compare to `cn_voi_shock` redundancy.  
**P3 Alt:** Concept note only until vendor feed is contracted.

---

## 4. Explicit non-goals this sprint

- More OHLCV “stability / reversal / shadow” variants  
- Re-tuning C2 weights (research freeze stands at 0.60/0.20/0.20)  
- Re-activating `value_composite` / `quality_composite` without industry-neutral rebuild  
- 100-factor grids inside one data source

---

## 5. Success definition (end of sprint)

| Outcome | Criteria |
|---------|----------|
| **Win** | ≥1 factor with residual IC t≳2 vs Base3, stack uplift at λ≤0.2, turnover fence OK → candidate enhancer or new dim |
| **Useful fail** | Clean negative: SUE absorbed or unstable by year → document and move to P1 |
| **Process win** | New-dimension loader + density template reusable for events/L2 |

---

## 6. Next action

**Immediate:** implement P0 Wind loaders + `sue_np_yoy_z` / `sue_eps_consensus` and run density vs Base3.  
Do not expand to P1 until P0 verdict JSON exists.
