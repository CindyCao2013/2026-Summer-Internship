// average_active_trade_size — DDB-native reference script

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

bars = select Symbol, Date, Bartime,
    iif(buy_count > 0, buy_amt \ buy_count, NULL) as buy_size
    from bars0
    context by Symbol, Date csort Bartime

features = select Symbol, Date, Bartime, buy_size,
    move(mavg(buy_size, 20, 10), 1) as hist_mean
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol,
    concatDateTime(Date, Bartime) as bartime,
    buy_size \ hist_mean - 1.0 as value
    from features
    where Bartime in btFilter
      and isValid(buy_size)
      and isValid(hist_mean)
      and hist_mean > 0

select Symbol, bartime, value from result
