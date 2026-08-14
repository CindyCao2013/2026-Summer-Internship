# Appendix — Reproduction Commands

Run from the repository root with the research environment:

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# 1. Build resumable strict daily primitives and lagged scales.
$PY l2_factor_reproduction/scripts/build_mid_trade_amount_normalized_cache.py \
  --stage primitives

# 2. Distribution-only calibration; this stage must not load returns.
$PY l2_factor_reproduction/scripts/build_mid_trade_amount_normalized_cache.py \
  --stage calibrate

# 3. Refuse to run unless frozen_config.json passes its hash checks.
$PY l2_factor_reproduction/scripts/build_mid_trade_amount_normalized_cache.py \
  --stage factors --workers 2

# 4. Enrich every resumable chunk with exact SQL/runtime lineage.
$PY l2_factor_reproduction/scripts/finalize_mid_trade_amount_normalized_lineage.py

# 5. Run the A0 panel hard gate and materialize the authoritative headline.
$PY l2_factor_reproduction/scripts/run_mid_trade_amount_normalized_parity.py \
  --stage panel

# 6. Generate standalone artifacts and all ten figure classes.
$PY l2_factor_reproduction/scripts/generate_mid_trade_amount_normalized_report.py

# 7. Render the six numeric Markdown chapters from persisted artifacts.
$PY l2_factor_reproduction/reporting/render_mid_trade_amount_normalized_markdown.py

# 8. Recheck A0 formal RankIC/ICIR/decile/H-L parity.
$PY l2_factor_reproduction/scripts/run_mid_trade_amount_normalized_parity.py \
  --stage formal

# 9. Export Markdown to embedded HTML and PDF, then hash the package.
$PY l2_factor_reproduction/scripts/export_mid_trade_amount_normalized_report.py
$PY l2_factor_reproduction/scripts/finalize_mid_trade_amount_normalized_package.py

# 10. Unit, regression, artifact, and package gates.
$PY -m pytest -q \
  l2_factor_reproduction/tests/test_mid_trade_amount_normalization.py \
  l2_factor_reproduction/tests/test_ch_mid_trade_amount_normalization.py \
  l2_factor_reproduction/tests/test_build_mid_trade_amount_normalized_cache.py \
  l2_factor_reproduction/tests/test_mid_trade_amount_normalized_report.py \
  l2_factor_reproduction/tests/test_render_mid_trade_amount_normalized_markdown.py \
  l2_factor_reproduction/tests/test_mid_trade_amount_normalized_artifacts.py \
  l2_factor_reproduction/tests/test_standalone_report_export.py \
  l2_factor_reproduction/tests/test_ch_tick.py \
  l2_factor_reproduction/tests/test_mid_order_ratio_report_artifacts.py \
  l2_factor_reproduction/tests/test_factor_dev_lib_metrics.py
```

The cache builder uses monthly resumable chunks.  Existing chunks are reused
unless an explicit force flag is supplied.  Stage B never changes the frozen
thresholds or direction.

