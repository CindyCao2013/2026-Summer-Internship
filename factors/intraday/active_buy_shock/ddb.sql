// active_buy_shock — Discovery v1 DDB reference

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date, second(Bartime) as Bartime,
    iif(isValid(Active_buy_amount) and Active_buy_amount > 0,
        Active_buy_amount, 0.0) as buy_amt
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

features = select Symbol, Date, Bartime, buy_amt,
    move(mavg(buy_amt, 20, 10), 1) as hist_mean,
    move(mstd(buy_amt, 20, 10), 1) as hist_std
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    (buy_amt - hist_mean) \ hist_std as value
    from features
    where Bartime in btFilter
      and isValid(hist_std)
      and hist_std > iif(abs(hist_mean) * 0.00000001 > 1.0,
        abs(hist_mean) * 0.00000001, 1.0)

select Symbol, bartime, value from result
