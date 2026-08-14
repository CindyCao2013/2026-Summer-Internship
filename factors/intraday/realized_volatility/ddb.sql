// realized_volatility — discovery_v1 DDB-native reference script
// Canonical builder: core.ddb_intraday_queries.discovery_v1_factor_script()

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars0 = select Symbol, Date, second(Bartime) as Bartime,
    Close as close_adj
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime,
    close_adj \ move(close_adj, 1) - 1.0 as minute_ret
    from bars0
    context by Symbol, Date csort Bartime

features = select Symbol, Date, Bartime,
    sqrt(cumsum(nullFill(minute_ret * minute_ret, 0.0))) as value,
    cumsum(iif(isValid(minute_ret), 1, 0)) as obs_count
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol, concatDateTime(Date, Bartime) as bartime, value
    from features
    where Bartime in btFilter and obs_count >= 5 and isValid(value)

select Symbol, bartime, value from result
