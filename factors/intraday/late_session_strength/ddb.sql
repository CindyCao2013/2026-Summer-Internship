// late_session_strength — DDB-native reference script
// Canonical builder: core.ddb_intraday_queries.late_session_strength_script()

rawStart = 2024.04.21
endDate = 2024.05.31

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date,
    Active_buy_amount
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as buy_amt,
    Active_sell_amount
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as sell_amt
    from t
    where Date between rawStart : endDate
      and second(Bartime) between 14:30:00 : 15:00:00

closingFlow = select
    nullFill(sum(buy_amt), 0) as buy_amt,
    nullFill(sum(sell_amt), 0) as sell_amt
    from bars
    group by Symbol, Date

result = select Symbol, Date,
    buy_amt \ (buy_amt + sell_amt) as value
    from closingFlow
    where buy_amt + sell_amt > 0

// Python preserves the existing BDay(T)+09:59 signal timestamp contract.
select Symbol, Date, value from result
