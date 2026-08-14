// intraday_amihud — discovery_v1 DDB reference

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars0 = select Symbol, Date, second(Bartime) as Bartime,
    Close as close_adj,
    iif(isValid(Amount) and Amount > 0, Amount, 0.0) as amount_adj
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime, amount_adj,
    close_adj \ move(close_adj, 1) - 1.0 as minute_ret
    from bars0
    context by Symbol, Date csort Bartime

features = select Symbol, Date, Bartime,
    msum(abs(minute_ret), 5, 3) as abs_ret_sum,
    msum(amount_adj, 5, 3) as amount_sum
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    abs_ret_sum \ amount_sum as value
    from features
    where Bartime in btFilter and amount_sum > 1.0

select Symbol, bartime, value from result
