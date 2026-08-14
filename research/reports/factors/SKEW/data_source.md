# SKEW — Data Source

| Item | Source |
|------|--------|
| Stock EOD | Wind `ASHAREEODPRICES` via `load_eod_enriched_tables` |
| Returns | close-to-close |
| Market | CSI300 `000300.SH` via `AINDEXEODPRICES` |
| Industry | CITICS industry panel |
| Size | float mktcap |
| TGD20 (interaction) | `research/cache/tgd_panels/TGD20_*_w20.parquet` |

P0 does **not** use minute bars. RSKEW20 is deferred (P1).
