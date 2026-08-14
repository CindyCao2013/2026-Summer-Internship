// active_buy_sell_imbalance — DDB-native reference script

startDate = 2024.05.01
endDate = 2024.05.31
btFilter = [09:59:00, 10:29:00, 11:29:00, 13:29:00, 14:29:00]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date, second(Bartime) as Bartime,
    nullFill(Active_buy_amount, 0)
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as buy_amt,
    nullFill(Active_sell_amount, 0)
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as sell_amt
    from t
    where Date between startDate : endDate
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

flows = select Symbol, Date, Bartime,
    cumsum(buy_amt) as cum_buy,
    cumsum(sell_amt) as cum_sell
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol,
    concatDateTime(Date, Bartime) as bartime,
    (cum_buy - cum_sell) \ (cum_buy + cum_sell) as value
    from flows
    where Bartime in btFilter
      and cum_buy + cum_sell > 0

select Symbol, bartime, value from result
