# 09 — In-sample / Out-of-sample Protocol

## Frozen chronology

- Distribution-only calibration: 2023-01-03 to 2023-06-30.
- Formal-sample comparability: return dates 2023-01-04 to 2024-06-28.
- Post-calibration validation: return dates 2023-07-03 to 2024-06-28.
- True OOS target: 2024-07-01 to the latest common complete month across
  strict Tick, Wind returns, tradability data, and PIT membership.

The target OOS endpoint is 2026-07-31 when all sources pass coverage checks.

## Frozen objects

Before any return evaluation, `frozen_config.json` records:

- A1 lower and upper bps-of-ADV bounds;
- A2 `0.5–2.0` bounds;
- candidate grids and the distribution-only A1 selection rule;
- effective direction `-1`;
- calibration dates;
- strict filter and source-cache hashes.

Stage B verifies the config checksum and refuses to continue if it changes.
Validation and OOS periods cannot select thresholds or infer direction again.

## Interpretation

The formal-sample table preserves comparability with the previous report.
Because A1 is selected without return information, it is not return-optimized;
nevertheless the post-calibration and true OOS tables are the stronger
evidence.  IS and OOS are reported separately and are never concatenated for
parameter selection.

## Execution limitation

The factor is complete only after the T-1 close.  T-1 signal versus T
close-to-close return includes the overnight component and is not a complete
execution simulation.  H-L results are fee-zero standalone sorting
diagnostics, not portfolio-strategy returns.

