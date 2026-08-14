# Dynamic Factor SQL Evidence

This is the exact first monthly query pair from the persisted dynamic cache.
Every monthly chunk metadata file stores its own SSE and SZSE SQL text and
SHA256. The execution used `clickhouse-connect`
`0.8.17` against ClickHouse
`23.2.3.17` with `CSVWithNames` ExternalData.

Chunk: `2023-01-03` to `2023-01-31`.

## SSE

```sql
WITH strict_ticks AS
    (
        
    SELECT
        Symbol,
        ExchTime,
        toFloat64(ifNull(Amount, Price * Volume)) AS amt
    FROM cmds.SSE_AL_TICK_EXG
    WHERE ExchTime >= toDateTime64('2023-01-03 09:30:00', 6, 'Asia/Shanghai')
      AND ExchTime <  toDateTime64('2023-01-31 15:00:01', 6, 'Asia/Shanghai')
      AND toDate(ExchTime) BETWEEN toDate('2023-01-03') AND toDate('2023-01-31')
      AND ((toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30) OR (toHour(ExchTime) > 9 AND toHour(ExchTime) < 15) OR (toHour(ExchTime) = 15 AND toMinute(ExchTime) = 0 AND toSecond(ExchTime) = 0))
      AND Type = 'T'
      AND Price > 0 AND Volume > 0
      AND startsWith(Symbol, '6')
      
    
    ),
    joined_ticks AS
    (
        SELECT
            t.Symbol AS Symbol,
            toDate(t.ExchTime) AS TradeDate,
            t.amt AS amt,
            s.ADV20_lag1 AS ADV20_lag1,
            s.ATS20_lag1 AS ATS20_lag1,
            s.q20 AS q20,
            s.q80 AS q80
        FROM strict_ticks AS t
        INNER JOIN scale_rows AS s
            ON t.Symbol = s.symbol
           AND toDate(t.ExchTime) = s.TradeDate
    )
    SELECT
        concat(Symbol, '.SH') AS symbol,
        TradeDate,
        sum(amt) AS total_amount,
        ifNull(sumIf(amt, amt > 40000 AND amt <= 200000), 0) AS `a0_abs_4w20w_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l0p5_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l0p5_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l0p5_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l1_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l1_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l1_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l2_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l2_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l2_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p25_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p25_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p25_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p5_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p5_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p5_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p75_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p75_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p75_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(q20) AND isNotNull(q80) AND q20 >= 0 AND q80 >= q20 AND amt > q20 AND amt <= q80), 0) AS `a3_q20_q80_selected_amount`
    FROM joined_ticks
    GROUP BY Symbol, TradeDate
    ORDER BY TradeDate, symbol
```

## SZSE

```sql
WITH strict_ticks AS
    (
        
    SELECT
        Symbol,
        ExchTime,
        toFloat64(Price * Volume) AS amt
    FROM cmds.SZSE_AL_TICK_EXG
    WHERE ExchTime >= toDateTime64('2023-01-03 09:30:00', 6, 'Asia/Shanghai')
      AND ExchTime <  toDateTime64('2023-01-31 15:00:01', 6, 'Asia/Shanghai')
      AND toDate(ExchTime) BETWEEN toDate('2023-01-03') AND toDate('2023-01-31')
      AND ((toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30) OR (toHour(ExchTime) > 9 AND toHour(ExchTime) < 15) OR (toHour(ExchTime) = 15 AND toMinute(ExchTime) = 0 AND toSecond(ExchTime) = 0))
      AND Type = '011' AND BidOrderNo > 0 AND AskOrderNo > 0
      AND Price > 0 AND Volume > 0
      AND (startsWith(Symbol, '000') OR startsWith(Symbol, '001') OR startsWith(Symbol, '002') OR startsWith(Symbol, '003') OR startsWith(Symbol, '300') OR startsWith(Symbol, '301') OR startsWith(Symbol, '302'))
      
    
    ),
    joined_ticks AS
    (
        SELECT
            t.Symbol AS Symbol,
            toDate(t.ExchTime) AS TradeDate,
            t.amt AS amt,
            s.ADV20_lag1 AS ADV20_lag1,
            s.ATS20_lag1 AS ATS20_lag1,
            s.q20 AS q20,
            s.q80 AS q80
        FROM strict_ticks AS t
        INNER JOIN scale_rows AS s
            ON t.Symbol = s.symbol
           AND toDate(t.ExchTime) = s.TradeDate
    )
    SELECT
        concat(Symbol, '.SZ') AS symbol,
        TradeDate,
        sum(amt) AS total_amount,
        ifNull(sumIf(amt, amt > 40000 AND amt <= 200000), 0) AS `a0_abs_4w20w_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l0p5_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l0p5_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 0.5 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l0p5_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l1_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l1_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 1 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l1_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 5), 0) AS `a1_adv20_bps_l2_h5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 10), 0) AS `a1_adv20_bps_l2_h10_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 AND (amt / ADV20_lag1 * 10000) > 2 AND (amt / ADV20_lag1 * 10000) <= 20), 0) AS `a1_adv20_bps_l2_h20_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p25_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p25_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.25 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p25_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p5_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p5_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.5 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p5_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 1.5), 0) AS `a2_ats20_l0p75_h1p5_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 2), 0) AS `a2_ats20_l0p75_h2_selected_amount`,
        ifNull(sumIf(amt, isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 AND (amt / ATS20_lag1) > 0.75 AND (amt / ATS20_lag1) <= 3), 0) AS `a2_ats20_l0p75_h3_selected_amount`,
        ifNull(sumIf(amt, isNotNull(q20) AND isNotNull(q80) AND q20 >= 0 AND q80 >= q20 AND amt > q20 AND amt <= q80), 0) AS `a3_q20_q80_selected_amount`
    FROM joined_ticks
    GROUP BY Symbol, TradeDate
    ORDER BY TradeDate, symbol
```
