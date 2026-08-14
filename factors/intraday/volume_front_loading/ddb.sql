// volume_front_loading — DDB-native reference script
// Canonical builder: core.ddb_intraday_queries.volume_front_loading_script()

startDate = 2024.05.01
endDate = 2024.05.31
histStart = 2024.04.01
lookback = 20
minP = 10

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
// pandas groupby.sum maps an all-null session to 0; nullFill preserves parity.
morning = select nullFill(sum(Volume), 0) as morning_vol
    from t
    where Date between histStart : endDate
      and second(Bartime) between 09:30:00 : 10:00:00
    group by Symbol, Date

morning = select Symbol, Date, morning_vol,
    // Shift the rolling sum/count by one session so today is excluded.
    move(msum(morning_vol, lookback, minP), 1)
        \ move(mcount(morning_vol, lookback, minP), 1) as hist_avg
    from morning
    context by Symbol csort Date

result = select Symbol,
    concatDateTime(Date, 10:29:00) as bartime,
    morning_vol \ hist_avg as value
    from morning
    where Date between startDate : endDate
      and isValid(hist_avg) and hist_avg > 0

select Symbol, bartime, value from result
