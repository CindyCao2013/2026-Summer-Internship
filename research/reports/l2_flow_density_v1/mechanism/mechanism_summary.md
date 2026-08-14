# FlowDensity20 Mechanism Validation v1

**Canonical factor:** `net_active_flow_mktcap_20d`
**Verdict:** `mechanism_entangled_with_anti_amount`
**Promote to validated_single_factor:** `False`
**Freeze formula:** `False`

## Research questions

1. Is alpha just an amount / volume effect?
2. Does net active flow beat buy/sell alone?
3. Does size / size+industry residual still carry alpha?

## Component ICIR (size+industry signal unless noted)

| Signal | Family | RankIC | ICIR | H-L Sharpe | Net@15bp | Note |
|--------|--------|-------:|-----:|-----------:|---------:|------|
| `active_buy_mktcap_20d` | component | -0.0476 | -8.73 | 3.84 | 3.43 | size+industry |
| `active_sell_mktcap_20d` | component | -0.0473 | -8.59 | 4.08 | 3.62 | size+industry |
| `net_active_flow_mktcap_20d` | canonical | 0.0236 | 4.85 | 3.38 | 1.92 | size+industry |
| `gross_active_mktcap_20d` | component | -0.0476 | -8.67 | 3.99 | 3.52 | size+industry |
| `amount_mktcap_20d` | component | -0.0475 | -8.66 | 3.95 | 3.55 | size+industry |
| `amount_eod_mktcap_20d` | component | -0.0474 | -8.66 | 3.69 | 3.56 | size+industry |
| `volume_mktcap_20d` | component | -0.0252 | -4.99 | 2.80 | 2.25 | size+industry |
| `active_buy_share_20d` | component | 0.0036 | 0.88 | 1.18 | -1.34 | size+industry |
| `net_size_resid` | style_residual | 0.0271 | 4.41 | 2.95 | 1.61 | Flow ⊥ size |
| `net_size_industry_resid` | style_residual | 0.0236 | 4.85 | 3.38 | 1.92 | Flow ⊥ size+industry (same as confirmation) |
| `net_active_flow_mktcap_20d|raw` | canonical_raw | 0.0178 | 2.07 | 1.52 | 0.11 | raw cs_z |
| `Flow_perp_Amount` | residual | -0.0058 | -1.66 | nan | nan | ⊥ L2 amount/mktcap 20d |
| `Flow_perp_AmountEOD` | residual | -0.0058 | -1.65 | nan | nan | ⊥ EOD amount/mktcap 20d |
| `Flow_perp_Volume` | residual | 0.0149 | 3.13 | nan | nan | ⊥ L2 volume/mktcap 20d |
| `Flow_perp_GrossActive` | residual | -0.0057 | -1.64 | nan | nan | ⊥ undirected gross active flow |
| `Flow_perp_Buy` | residual | -0.0036 | -1.02 | nan | nan | ⊥ buy leg |
| `Flow_perp_Sell` | residual | -0.0076 | -2.21 | nan | nan | ⊥ sell leg |

## Gate checklist

- direction ≠ undirected activity (net>0, amount/gross/buy≪0): **True**
- net distinct from buy/sell legs (sign flip): **True**
- residual POSITIVE vs Amount (strict freeze gate): **False** (ICIR=-1.66, corr=-0.617)
- residual POSITIVE vs Volume: **True** (ICIR=3.13)
- residual POSITIVE vs GrossActive: **False** (ICIR=-1.64)
- entangled with anti-amount: **True**

## Interpretation

Net flow is NOT undirected activity: buy/sell/gross/amount all have large NEGATIVE ICIR (amount=-8.66, gross=-8.67) while net has POSITIVE ICIR (4.85). However Flow⊥Amount residual ICIR=-1.66 (cs_corr=-0.617) — positive net alpha is entangled with anti-amount / low-activity exposure. Do NOT freeze yet; consider amount-orthogonalized net or document liquidity entanglement before validated_single_factor.

## Artifacts

- `mechanism.csv` — canonical pack file
- `mechanism_components.csv` — full eval table
- `mechanism_residuals.csv` — residual IC table
- `mechanism_verdict.json`

## Next (if entangled)

1. Build `net_active_flow_mktcap_20d ⊥ amount_mktcap_20d` as candidate cleaner signal
2. Or document liquidity entanglement and keep combination-only use
3. Then TGD ⟂ Flow orthogonality

