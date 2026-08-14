# [SIDETRACK / Phase 3 preview] Portfolio Construction Layer v1

**Date:** 2026-07-20  
**Status:** PARKED — not the active OS milestone  
**Official Milestone 2.2:** [`docs/milestone_2_2_research_track_expansion.md`](milestone_2_2_research_track_expansion.md)  
**Script:** `run_portfolio_construction_v1.py` (archived experiment)  
**Outputs:** `research/reports/portfolio_construction_v1/`

> **Capability complete / FROZEN.** This proves portfolio *capability* (sizing · risk · capacity),  
> not Portfolio Production. Mainline is **Phase III Alpha Library Expansion**  
> ([`docs/milestone_3_0_alpha_library_expansion.md`](milestone_3_0_alpha_library_expansion.md)).  
> Reopen Portfolio Construction v2 only after ~20–30 comparable research assets.

---

## Original intent (parked)

```
Factor Research OS  →  (premature) Portfolio Alpha Construction
```

This sidetrack did **not** add factors. It experimented with sizing / risk / capacity on B/C.

---

## Frozen Alpha Information Topology v1

| Factor | Role | Function |
|--------|------|----------|
| TGD20 | `source` | Alpha generation |
| D1 | `stabilizer` | Alpha stabilization |
| FlowDensity20 | `enhancer` | Portfolio enhancement (not 3rd IC source) |

**Strategy identity:** China A-share Mid/Small Cap Microstructure Alpha  
**Engine:** B = TGD+D1 baseline · C = B+Flow optional enhancer  
**Registry:** `alpha_role` proposed only — schema **not** modified (constraint).

---

## Window / book

2020-01-02 → 2025-12-31 (1455d) · SI CS-z · top_frac=0.10 · 15bp RT

---

## 1. Position sizing

| Model | Combine | Exposure | Net Sharpe | Daily TO | MDD net |
|-------|---------|----------|----------:|---------:|--------:|
| B | equal | none | 2.21 | 0.488 | −0.115 |
| C | equal | none | **2.92** | 0.465 | −0.085 |
| B | ic_weighted | none | 2.16 | 0.480 | −0.118 |
| C | ic_weighted | none | 2.62 | 0.465 | −0.094 |
| B | ic_weighted | vol_scaled | 2.14 | 0.704 | −0.161 |
| C | ic_weighted | vol_scaled | 2.65 | 0.711 | −0.135 |

**Takeaway:** Equal vs IC combine is robust (B≈2.1–2.2). Vol scaling lifts ann. return via leverage but worsens MDD / turnover — not free alpha.

---

## 2. Risk controls (IC-weighted)

| Model | Mode | Net Sharpe | MDD | Mean exp |
|-------|------|----------:|----:|---------:|
| B | none | 2.16 | −0.118 | 1.00 |
| B | vol_target (15% ann) | 2.14 | −0.161 | 1.42 |
| B | dd_control (−10%) | 2.16 | −0.118 | 1.00 |
| B | vol_dd | 2.12 | −0.160 | 1.41 |
| C | none | 2.62 | −0.094 | 1.00 |
| C | vol_target | **2.65** | −0.135 | 1.48 |
| C | dd_control | 2.62 | −0.094 | 1.00 |
| C | vol_dd | 2.64 | −0.135 | 1.48 |

**Takeaway:**  
- Vol targeting ≈ leverage tool (mean exp ~1.4–1.5); Sharpe flat, MDD deeper.  
- DD@−10% on gross path **not binding** this sample (net MDD still ~9–12%). Tighten threshold or apply on net equity if production needs active risk-off.  
- Do **not** promote vol overlays as alpha — they are risk packaging.

---

## 3. Capacity diagnostics

| Model | Approx capacity (5% ADV) | @100M part. | @500M part. | Within 5% @500M |
|-------|-------------------------:|------------:|------------:|:---------------:|
| B | ~¥9.4bn | 0.025% | 0.13% | ✅ |
| C | ~¥9.2bn | 0.025% | 0.13% | ✅ |

**Caveat (important):** Capacity uses **mean book ADV × n_names × 5%** — an **upper bound**. True binding constraint is thin names in the mid/small book (p10 ADV / single-name caps). Treat ¥9bn as optimistic ceiling, not deployable AUM.

For 10–500M CNY research books, participation diagnostics are comfortable under this mean-ADV model; production must add **per-name ADV caps** next.

---

## 4. Regime — when does Flow enhance?

Δ Net Sharpe (C − B), IC-weighted, no overlay:

| Regime | ΔNet Sharpe |
|--------|------------:|
| bull | **+0.62** |
| bear | +0.28 |
| sideways | +0.36 |
| high_vol | +0.51 |
| low_vol | +0.47 |

**Takeaway:** Enhancer value is **broad**, strongest in **bull / high-vol**. Still not a new alpha source — conditional implementation lift.

---

## Verdict

| Decision | Choice |
|----------|--------|
| Production baseline | **B (TGD+D1)**, IC or equal combine |
| Optional overlay | **C** (Flow enhancer) |
| Default risk package | `exposure=none` until DD rule is retuned |
| Vol targeting | Optional packaging only |
| Capital (diagnostic) | 10–500M OK under mean-ADV; need name-level caps before scale-up |
| Next information layer | Fundamental **after** name-level capacity / buffer execution |

---

## Artifacts

| File | Content |
|------|---------|
| `position_sizing.csv` | equal / IC / vol-scaled |
| `risk_controls.csv` | none · vol_target · dd · vol_dd |
| `capacity_diagnostics.csv` | capital grid × participation |
| `regime_B.csv` / `regime_C.csv` / `regime_enhancer_delta.csv` | regime nets |
| `portfolio_report.md` | human report |
| `portfolio_verdict.json` | machine summary |
| `charts/*.png` | NAV · capacity |

---

## Constraints honored

- No new factors  
- No TGD/D1/Flow formula changes  
- No Registry schema edits  

---

## Next

```
2.2 Portfolio Construction     ✅
        ↓
2.2.x Name-level ADV caps + buffer execution (optional harden)
        ↓
2.3 / Phase III  Fundamental Alpha Layer
```
