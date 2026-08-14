// minute_skew — DDB-native reference script (abridged production formula)
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

moments = select Symbol, Date, Bartime,
    cumsum(iif(isValid(minute_ret), 1.0, 0.0)) as n,
    cumsum(nullFill(minute_ret, 0.0)) as s1,
    cumsum(nullFill(minute_ret * minute_ret, 0.0)) as s2,
    cumsum(nullFill(minute_ret * minute_ret * minute_ret, 0.0)) as s3
    from bars
    context by Symbol, Date csort Bartime

central = select Symbol, Date, Bartime, n,
    s2 - s1 * s1 \ n as m2,
    s3 - 3.0 * s1 * s2 \ n + 2.0 * s1 * s1 * s1 \ (n * n) as m3
    from moments
    where n >= 3

result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    n * sqrt(n - 1.0) \ (n - 2.0) * m3 \ pow(m2, 1.5) as value
    from central
    where Bartime in btFilter and m2 > 0

select Symbol, bartime, value from result
