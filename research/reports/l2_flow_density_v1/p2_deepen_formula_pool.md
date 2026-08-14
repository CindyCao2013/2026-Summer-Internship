# P2 deepen — formula pool (post-confirmation)

Run these **only after** `net_active_flow_mktcap_20d` confirmation + investability passes.
Reuse `run_l2_flow_density_v1.py` / density gates; do not retune the confirmed 20d formula.

| Formula ID | Logic | Diff vs `net_active_flow_mktcap_20d` |
|------------|-------|--------------------------------------|
| `net_active_flow_mktcap_5d` | 5d cum active net flow / float mktcap | Already in discovery (ICIR 3.62); shorter horizon, higher TO |
| `large_order_imbalance_20d` | (large buy − large sell) / total amt, 20d | Smart-money filter vs all-size active flow |
| `open_auction_imbalance_20d` | Open-auction B/S imbalance, 20d cum | Auction microstructure ≠ continuous session flow |

**Gate (same as P2 discovery):** size+industry neut → residual_t vs Base3 ≥ 2 → stack ICIR uplift > 0 → TO acceptable. Prefer residual vs `cn_voi_shock` as well so new formulas are not VOI clones.
