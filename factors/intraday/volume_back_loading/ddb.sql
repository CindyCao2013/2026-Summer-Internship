// volume_back_loading — DDB-native reference script
// Canonical builder: core.ddb_intraday_queries.volume_back_loading_script()

startDate = 2024.05.01
endDate = 2024.05.31
histStart = 2024.04.01
lookback = 20
minP = 10

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
// nullFill matches pandas groupby.sum for an all-null session.
closing = select nullFill(sum(Volume), 0) as closing_vol
    from t
    where Date between histStart : endDate
      and second(Bartime) between 14:30:00 : 15:00:00
    group by Symbol, Date

closing = select Symbol, Date, closing_vol,
    // Shift one session: today's numerator is excluded from its denominator.
    move(msum(closing_vol, lookback, minP), 1)
        \ move(mcount(closing_vol, lookback, minP), 1) as hist_avg
    from closing
    context by Symbol csort Date

result = select Symbol, Date,
    closing_vol \ hist_avg as value
    from closing
    where Date between startDate : endDate
      and isValid(hist_avg) and hist_avg > 0

// Python applies the existing BDay(T)+09:59 timestamp contract to this narrow result.
select Symbol, Date, value from result
