// ofi_persistence — representative output of discovery_v1_factor_script

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars0 = select Symbol, Date, second(Bartime) as Bartime,
    iif(isValid(Active_buy_amount) and Active_buy_amount > 0,
        Active_buy_amount, 0.0) as buy_amt,
    iif(isValid(Active_sell_amount) and Active_sell_amount > 0,
        Active_sell_amount, 0.0) as sell_amt,
    iif(isValid(Active_buy_count) and Active_buy_count > 0,
        Active_buy_count, 0) as buy_count,
    iif(isValid(Active_sell_count) and Active_sell_count > 0,
        Active_sell_count, 0) as sell_count,
    Close as close_adj,
    iif(isValid(Amount) and Amount > 0, Amount, 0.0) as amount_adj
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime, buy_amt, sell_amt, buy_count, sell_count,
    close_adj, amount_adj,
    iif(buy_amt + sell_amt > 0,
        (buy_amt - sell_amt) \ (buy_amt + sell_amt), NULL) as bar_ofi,
    iif(buy_count > 0, buy_amt \ buy_count, NULL) as buy_size,
    close_adj \ move(close_adj, 1) - 1.0 as minute_ret
    from bars0
    context by Symbol, Date csort Bartime

features = select Symbol, Date, Bartime,
    msum(iif(isValid(bar_ofi), iif(bar_ofi > 0, 1.0, 0.0), NULL), 20, 5)
        \ mcount(bar_ofi, 20, 5) as value
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol, concatDateTime(Date, Bartime) as bartime, value
    from features
    where Bartime in btFilter and isValid(value)

select Symbol, bartime, value from result
