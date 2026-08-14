// close_vwap_deviation — DDB-native batch script (reference copy)
// Canonical builder: core/ddb_intraday_queries.close_vwap_deviation_script()
// Requirements: loadTable, Date partition filter, context by, server-side agg, narrow output

startDate = 2024.01.01
endDate = 2024.01.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date, second(Bartime) as Bartime, Close, Volume, Amount, Adjfactor
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime,
    Close * Adjfactor as close_adj,
    Volume,
    cumsum(Amount * Adjfactor) \ cumsum(Volume) as cum_vwap,
    rowNo(Bartime) as bar_idx
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol,
    concatDateTime(Date, Bartime) as bartime,
    (close_adj - cum_vwap) \ cum_vwap as value
    from bars
    where Bartime in btFilter
      and Volume > 0
      and isValid(cum_vwap) and cum_vwap != 0
      and bar_idx > 0

select Symbol, bartime, value from result
