# Portfolio Construction Layer v1

**Window:** 2020-01-02 → 2025-12-31 (1455d)
**Universe identity:** China A-share Mid/Small Cap Microstructure Alpha
**Book:** top_frac=0.1 · cost=0.0015 RT · SI CS-z

## Frozen roles

| Factor | Role |
|--------|------|
| TGD20 | primary alpha source (generation) |
| D1 | stabilizer / independent source |
| FlowDensity20 | combination enhancer |

See `docs/alpha_information_topology_v1.md`.

## 1. Position sizing (no risk overlay)

| model | combine | net_sharpe | gross_sharpe | mdd_net | daily_turnover | mean_exposure |
| --- | --- | --- | --- | --- | --- | --- |
| B | equal | 2.214 | 3.768 | -0.115 | 0.488 | 1.000 |
| C | equal | 2.921 | 4.559 | -0.085 | 0.465 | 1.000 |
| B | ic_weighted | 2.157 | 3.670 | -0.118 | 0.480 | 1.000 |
| C | ic_weighted | 2.617 | 4.175 | -0.094 | 0.465 | 1.000 |
| B | ic_weighted | 2.140 | 3.808 | -0.161 | 0.704 | 1.417 |
| C | ic_weighted | 2.654 | 4.372 | -0.135 | 0.711 | 1.479 |

## 2. Risk controls (IC-weighted combine)

| model | exposure_mode | net_sharpe | gross_sharpe | mdd_net | daily_turnover | mean_exposure | p95_exposure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | none | 2.157 | 3.670 | -0.118 | 0.480 | 1.000 | 1.000 |
| B | vol_target | 2.140 | 3.808 | -0.161 | 0.704 | 1.417 | 2.191 |
| B | dd_control | 2.157 | 3.670 | -0.118 | 0.480 | 1.000 | 1.000 |
| B | vol_dd | 2.122 | 3.801 | -0.160 | 0.705 | 1.413 | 2.191 |
| C | none | 2.617 | 4.175 | -0.094 | 0.465 | 1.000 | 1.000 |
| C | vol_target | 2.654 | 4.372 | -0.135 | 0.711 | 1.479 | 2.258 |
| C | dd_control | 2.617 | 4.175 | -0.094 | 0.465 | 1.000 | 1.000 |
| C | vol_dd | 2.644 | 4.365 | -0.135 | 0.712 | 1.479 | 2.258 |

## 3. Capacity (IC-weighted · exposure=none)

| model | capital_m | capacity_cny_approx | adv_participation | within_cap | daily_turnover |
| --- | --- | --- | --- | --- | --- |
| B | 10.000 | 9427756499.025 | 0.000 | True | 0.480 |
| B | 50.000 | 9427756499.025 | 0.000 | True | 0.480 |
| B | 100.000 | 9427756499.025 | 0.000 | True | 0.480 |
| B | 500.000 | 9427756499.025 | 0.001 | True | 0.480 |
| C | 10.000 | 9203393194.125 | 0.000 | True | 0.465 |
| C | 50.000 | 9203393194.125 | 0.000 | True | 0.465 |
| C | 100.000 | 9203393194.125 | 0.000 | True | 0.465 |
| C | 500.000 | 9203393194.125 | 0.001 | True | 0.465 |

Participation cap = 5% of book ADV.

## 4. Regime (IC-weighted · none) — Flow when?

### Model B

| regime | n_days | net_sharpe |
| --- | --- | --- |
| bull | 512 | 1.031 |
| bear | 146 | 4.884 |
| sideways | 797 | 2.403 |
| high_vol | 713 | 2.531 |
| low_vol | 713 | 1.878 |

### Model C

| regime | n_days | net_sharpe |
| --- | --- | --- |
| bull | 512 | 1.646 |
| bear | 146 | 5.160 |
| sideways | 797 | 2.763 |
| high_vol | 713 | 3.037 |
| low_vol | 713 | 2.351 |

## Interpretation

- Prefer **B** as production baseline; **C** as optional enhancer.
- Vol targeting / DD control trade Sharpe vs MDD — report, do not auto-promote.
- Capacity: mid/small microstructure → capital ceiling is binding before CSI300 universe is.
- No new factors. Registry schema unchanged.

