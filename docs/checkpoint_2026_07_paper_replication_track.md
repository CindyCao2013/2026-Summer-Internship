# Checkpoint — Alpha Research OS · Paper Replication Track (2026-07)

**Status:** Phase I infrastructure closed · **III-A CLOSED** · **III-B OPEN**  
**Superseding close-out:** [`docs/checkpoint_2026_07_iiia_complete.md`](checkpoint_2026_07_iiia_complete.md)  
**Portfolio / Composite:** capability validated, **frozen** (not production)

---

## One-line verdict

> Infrastructure + microstructure family (incl. SmartMoney research_candidate) are in place. Bottleneck is now **fundamental coverage**. Next: **III-B SUE** — only then Similarity/Composite/Portfolio **v2**.

---

## Layer status

| Layer | Status | Note |
|-------|--------|------|
| Paper → hypothesis → object/knife/output → spec | ✅ | Cutting DSL |
| Factor Research Protocol v1 | ✅ | |
| Report Generator Template v2 | ✅ | Golden + representative packs |
| factor_spec → harness → pack → Registry | ✅ | Factory path |
| Microstructure family | 🟢 main done | TGD / Flow / D1 / Cutting Ideal* |
| Fundamental family | 🔴 not started | SUE / value / quality |
| DL enhancement (DBD-GRU) | ⏸ paused | After classical validated |
| Batch paper replication | 🟡 ready | Waiting queue |
| Portfolio Construction | ✅ capability / **FROZEN** | Future consumer |

---

## Library map (honest)

```
Microstructure Family
├── Temporal      TGD20              validated
├── Liquidity     D1                 candidate
├── Flow          FlowDensity20      candidate (enhancer, not core)
└── Cutting
    ├── IdealReversal                testing (mono weak)
    ├── IdealAmplitude               testing (mono weak; high Sharpe)
    ├── ActiveTrade / APM            ← A3: daily PROXY only (paper needs minute)
    └── SmartMoney                   stub (minute)
```

| Factor | Role | Status |
|--------|------|--------|
| TGD20 | Core temporal | validated |
| D1 | Independent liquidity | candidate |
| FlowDensity20 | Satellite enhancer | candidate |
| IdealReversal | Cutting research | testing |
| IdealAmplitude | Cutting research | testing |

**Lesson (IdealAmplitude):** high Sharpe ≠ admit — mono / payoff shape matters.

---

## Phase III structure (locked)

```
III-A  Microstructure completion
       IdealReversal ✅ · IdealAmplitude ✅ · ActiveTrade(proxy) · SmartMoney(later)

III-B  Fundamental
       SUE → Revision → Value → Quality

III-C  Alternative
       L2 / Event / ML enhancement (after classical)

THEN   Similarity Matrix v2 → Composite v2 → Portfolio Construction v2
```

**Reopen Portfolio v2 only when** ~7–10 microstructure + ~5–10 fundamental (+ optional alt) exist as comparable packs.

---

## ActiveTrade / APM honesty gate

Paper 《主动买卖 / APM》 needs **session/minute** residuals.  
Repo ships `compute_apm_overnight_day_proxy` — **NOT paper APM**.

A3 admits proxy as `testing` with explicit `research_proxy` label.  
Do **not** promote to candidate as if paper-replicated.

---

## Related

- Expansion: `docs/milestone_3_0_alpha_library_expansion.md`
- Topology: `docs/alpha_information_topology_v1.md`
- Parked portfolio: `docs/milestone_2_2_portfolio_construction.md`
