# APM_SessionResidual — Formula

**Version:** `apm_session_v1_adapted_eod_index` · **frozen**

## 1. Stock session returns

\[
r^{\mathrm{ON}}_{i,t}=\frac{\mathrm{Open}_{i,t}}{\mathrm{Close}_{i,t-1}}-1
\]

\[
r^{\mathrm{PM}}_{i,t}=\frac{\mathrm{Close}^{\mathrm{last\,PM}}_{i,t}}{\mathrm{Close}^{\mathrm{first\,available\,PM}}_{i,t}}-1
\]

PM window: `Bartime ∈ [13:01, 15:00]`.  
`pm_start_rule = first_available_bar_after_13_01` (≥2 bars required).

## 2. Index adapter (adapted)

\[
r^{\mathrm{ON}}_{\mathrm{idx},t}=\frac{\mathrm{IdxOpen}_t}{\mathrm{IdxClose}_{t-1}}-1
\]

\[
r^{\mathrm{DAY}}_{\mathrm{idx},t}=\frac{\mathrm{IdxClose}_t}{\mathrm{IdxOpen}_t}-1
\quad\text{(full daytime proxy — not PM-matched)}
\]

Default index: `000852.SH`.

## 3. Residual legs

\[
\alpha^{\mathrm{ON}}_{i,t}=r^{\mathrm{ON}}_{i,t}-r^{\mathrm{ON}}_{\mathrm{idx},t},\quad
\alpha^{\mathrm{PM}}_{i,t}=r^{\mathrm{PM}}_{i,t}-r^{\mathrm{DAY}}_{\mathrm{idx},t}
\]

\[
\delta_{i,t}=\alpha^{\mathrm{ON}}_{i,t}-\alpha^{\mathrm{PM}}_{i,t}
\]

## 4. Rolling APM_stat

\[
\mathrm{APM\_stat}_{i,t}
=\frac{\mathrm{mean}(\delta_{i,t-19:t})}{\mathrm{std}(\delta_{i,t-19:t})/\sqrt{n}},\quad
\min\_periods=10
\]

## 5. Evaluation signal `apm_cs`

Per date, CS OLS residual of `APM_stat` on Ret20 (+ intercept).

Cache stores **unshifted** `apm_cs(T)`. Evaluation uses `shift(1)` → `return(T+1)`.

## 6. Not the formula

| Alias | Status |
|-------|--------|
| `ActiveTradeProxy` ON−DAY t-stat | different factor |
| `(ActiveBuy−ActiveSell)/(Buy+Sell)` | forbidden under this id |
| SmartMoney \(S_t=\|R\|/V^{0.25}\) | different paper |
