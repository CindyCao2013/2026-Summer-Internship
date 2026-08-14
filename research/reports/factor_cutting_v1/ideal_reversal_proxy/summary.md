# ideal_reversal — Proxy Knife Full History

**Label:** `ideal_reversal_proxy_amount`

NOT paper ATS on full history. ATS used where trade_count coverage>=200; else `amount`.

- Best proxy: `amount` · IC-series corr vs ATS = **0.757** (BELOW threshold 0.8)
- Full sample RankIC **-0.0281** · ICIR **-4.81** · monthly **-0.0563**
- Overlap ATS-only RankIC -0.0338 (reference)
- ATS days (coverage≥200): 1723 · proxy-fill days in sample: 2163

## Yearly RankIC (stitched)

 year   n   rank_ic      icir
 2010 241 -0.015266 -2.359740
 2011 244 -0.016441 -3.031443
 2012 243 -0.021614 -3.697376
 2013 238 -0.034682 -6.026837
 2014 245 -0.020951 -3.378001
 2015 244 -0.010834 -1.234589
 2016 244 -0.035990 -5.958733
 2017 244 -0.025816 -4.176833
 2018 224 -0.033461 -6.563701
 2019 244 -0.040626 -8.943156
 2020 243 -0.022919 -4.867478
 2021 243 -0.029457 -5.434293
 2022 242 -0.029332 -6.698366
 2023 242 -0.043477 -9.164986
 2024 242 -0.025122 -3.711697
 2025 243 -0.044893 -8.276743

## Neutralization

         mode   rank_ic      icir  ic_pos_ratio  n_days  monthly_rank_ic  monthly_icir  ic_retention_vs_raw
          raw -0.028142 -4.805637      0.348939    3866        -0.056263     -2.300471             1.000000
         size -0.033515 -6.823437      0.315297    1922        -0.063086     -2.828662             1.190919
     industry -0.029624 -7.816728      0.284044    1454        -0.053389     -2.915420             1.052669
size_industry -0.029466 -8.652438      0.273040    1454        -0.052913     -3.328831             1.047039

See `proxy_match_report.md` for knife ranking.
