// large_active_buy_ratio — bar-level proxy, not a true large-order ratio
// Canonical builder: core.ddb_intraday_queries.discovery_v1_factor_script()

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars0 = select Symbol, Date, second(Bartime) as Bartime,
    iif(isValid(Active_buy_amount) and Active_buy_amount > 0,
        Active_buy_amount, 0.0) as buy_amt,
    iif(isValid(Active_buy_count) and Active_buy_count > 0,
        Active_buy_count, 0) as buy_count
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime, buy_amt,
    iif(buy_count > 0, buy_amt \ buy_count, NULL) as buy_size
    from bars0
    context by Symbol, Date csort Bartime

baseline = select Symbol, Date, Bartime, buy_amt, buy_size,
    // Positive move shifts both statistics: the current bar is excluded.
    move(mavg(buy_size, 20, 10), 1) as hist_mean,
    // mstd is the prior-window sample standard deviation.
    move(mstd(buy_size, 20, 10), 1) as hist_std
    from bars
    context by Symbol, Date csort Bartime

classified = select Symbol, Date, Bartime, buy_amt,
    iif(isValid(buy_size) and isValid(hist_mean) and isValid(hist_std)
        and buy_size > hist_mean + hist_std, buy_amt, 0.0) as large_buy_amt,
    iif(isValid(hist_mean) and isValid(hist_std), 1.0, NULL) as baseline_valid
    from baseline

features = select Symbol, Date, Bartime,
    msum(large_buy_amt, 20, 10) as large_buy_sum,
    msum(buy_amt, 20, 10) as buy_sum,
    mcount(baseline_valid, 20, 10) as valid_count
    from classified
    context by Symbol, Date csort Bartime

result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    large_buy_sum \ buy_sum as value
    from features
    where Bartime in btFilter and valid_count >= 10
      and buy_sum > 1.0

select Symbol, bartime, value from result
