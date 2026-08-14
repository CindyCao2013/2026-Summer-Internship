# Composite Production Stress v1 (Milestone 2.1)

**Window:** 2020-01-02 → 2025-12-31 (1455d)
**Book:** size+industry CS-z · top_frac=0.1 · baseline cost 15bp

## Alpha roles (locked)

| Role | Factor |
|------|--------|
| Primary alpha source | TGD20 |
| Independent alpha source | D1 |
| Combination enhancer | FlowDensity20 |

Net Sharpe stacking ≠ three cores.

## A. Universe robustness

| universe | model | rank_icir | gross_sharpe | net_sharpe | daily_turnover | n_days |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | A | 11.166 | 4.363 | 1.376 | 0.665 | 1454 |
| ALL | B | 10.433 | 3.670 | 2.157 | 0.480 | 1454 |
| ALL | C | 10.385 | 4.175 | 2.617 | 0.465 | 1454 |
| CSI300 | A | 3.684 | 1.557 | -0.377 | 0.701 | 1454 |
| CSI300 | B | 3.523 | 0.703 | -0.635 | 0.552 | 1454 |
| CSI300 | C | 3.088 | 0.630 | -0.832 | 0.541 | 1454 |
| CSI500 | A | 4.941 | 1.581 | -0.650 | 0.685 | 1454 |
| CSI500 | B | 5.360 | 0.746 | -0.582 | 0.518 | 1454 |
| CSI500 | C | 5.092 | 1.051 | -0.371 | 0.507 | 1454 |
| CSI1000 | A | 7.745 | 3.214 | 0.691 | 0.678 | 1454 |
| CSI1000 | B | 7.724 | 2.408 | 0.992 | 0.503 | 1454 |
| CSI1000 | C | 7.475 | 2.796 | 1.328 | 0.492 | 1454 |

## B. Cost sensitivity (ALL · rolling_ic_60)

| model | cost_bp | net_sharpe | gross_sharpe | daily_turnover | mdd_net |
| --- | --- | --- | --- | --- | --- |
| A | 10 | 2.372 | 4.363 | 0.665 | -0.076 |
| B | 10 | 2.661 | 3.670 | 0.480 | -0.111 |
| C | 10 | 3.136 | 4.175 | 0.465 | -0.087 |
| A | 15 | 1.376 | 4.363 | 0.665 | -0.084 |
| B | 15 | 2.157 | 3.670 | 0.480 | -0.118 |
| C | 15 | 2.617 | 4.175 | 0.465 | -0.094 |
| A | 20 | 0.380 | 4.363 | 0.665 | -0.164 |
| B | 20 | 1.653 | 3.670 | 0.480 | -0.131 |
| C | 20 | 2.097 | 4.175 | 0.465 | -0.101 |
| A | 30 | -1.611 | 4.363 | 0.665 | -0.554 |
| B | 30 | 0.645 | 3.670 | 0.480 | -0.203 |
| C | 30 | 1.056 | 4.175 | 0.465 | -0.163 |
| A | 50 | -5.577 | 4.363 | 0.665 | -0.936 |
| B | 50 | -1.369 | 3.670 | 0.480 | -0.632 |
| C | 50 | -1.022 | 4.175 | 0.465 | -0.527 |

## C. Weight robustness (ALL · 15bp)

| scheme | model | rank_icir | net_sharpe | gross_sharpe | daily_turnover |
| --- | --- | --- | --- | --- | --- |
| static_50_50 | A | 11.166 | 1.376 | 4.363 | 0.665 |
| static_50_50 | B | 10.739 | 2.214 | 3.768 | 0.488 |
| static_50_50 | C | 10.275 | 2.921 | 4.559 | 0.465 |
| rolling_ic_60 | A | 11.166 | 1.376 | 4.363 | 0.665 |
| rolling_ic_60 | B | 10.433 | 2.157 | 3.670 | 0.480 |
| rolling_ic_60 | C | 10.385 | 2.617 | 4.175 | 0.465 |
| rolling_ic_120 | A | 11.166 | 1.376 | 4.363 | 0.665 |
| rolling_ic_120 | B | 10.423 | 2.178 | 3.684 | 0.479 |
| rolling_ic_120 | C | 10.475 | 2.747 | 4.306 | 0.461 |
| vol_adj_ic_60 | A | 11.166 | 1.376 | 4.363 | 0.665 |
| vol_adj_ic_60 | B | 10.855 | 2.146 | 3.716 | 0.493 |
| vol_adj_ic_60 | C | 10.742 | 2.550 | 4.160 | 0.481 |

## D. Calendar OOS (ALL · rolling_ic_60 · 15bp)

Note: discovery starts at data `2020-01-02` (cfg.START_DAY), not 2018.

| period | model | rank_icir | net_sharpe | gross_sharpe | daily_turnover | n_days |
| --- | --- | --- | --- | --- | --- | --- |
| discovery | A | 11.568 | 1.644 | 5.071 | 0.683 | 727 |
| discovery | B | 10.424 | 2.160 | 3.782 | 0.494 | 727 |
| discovery | C | 10.287 | 2.536 | 4.142 | 0.482 | 727 |
| validation | A | 11.393 | 1.777 | 4.455 | 0.638 | 483 |
| validation | B | 11.538 | 2.846 | 4.275 | 0.472 | 483 |
| validation | C | 11.576 | 3.188 | 4.753 | 0.456 | 483 |
| test | A | 10.295 | 0.129 | 2.780 | 0.672 | 242 |
| test | B | 9.212 | 0.924 | 2.351 | 0.463 | 242 |
| test | C | 9.230 | 1.952 | 3.405 | 0.443 | 242 |

## Stress verdict

- ALL B Net Sharpe=2.16; CSI1000 B Net=0.99 (pass: not mega-cap-only).
- At 50bp RT: B Net=-1.37, C Net=-1.02 (breaks under high cost).
- Test period: B Net=0.92, C Net=1.95 (OOS positive).

## Next

- If stress passes: freeze B as production baseline candidate; C optional overlay.
- If fails on CSI1000 or Test: diagnose size/regime before Fundamental layer.
- Do **not** expand D4/D5 into composite until stress is clean.
- Next research layer: Fundamental / Value / Risk (information expansion).

