# Factor Registry v1

Canonical inventory for Alpha Research OS.

| File | Role |
|------|------|
| `factor_registry.yaml` | Full structured registry (source of truth for nested fields) |
| `factor_registry.csv` | Flat spreadsheet view for PM / inventory scans |

**Rules**

1. One `factor_id` = one economic hypothesis + one investable expression.  
2. Diagnostics (`εd`, `τ`, `M_high`, …) and execution labels (`buffer_5_15`, …) **must not** appear as rows.  
3. Status follows Protocol: `discovery | testing | candidate | validated | production | retired`.  
4. Admission always `requires_manual_review` (no numeric auto-gate).  
5. `correlation_cluster` stays null until Factor Similarity Matrix (1F).

See `docs/milestone_1E_registry_v1.md`.
