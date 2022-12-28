"""
▪ Many banks now consider that OIS rates should be used for
discounting when collateralized portfolios are valued and that
LIBOR should be used for discounting when portfolios are not
collateralized.

https://www.linkedin.com/pulse/python-bootstrapping-zero-curve-sheikh-pancham/
This is called the settlement date and all cash flows are
net present valued to the settlement date when the deal is struck.
So given a curve date of today, the curve settlement date is calculated
as the number of settle days from today, adjusted to business days.
"""

import numpy as np
import pandas as pd

from idxdata.historical_data import get_price_from_sql

import xlwings as xw
import QuantLib as ql
import matplotlib.pyplot as plt

from datetime import date, timedelta


def get_KRWIRSdata():

    Name_rates = [
        "KW CD91",
        "KW SWAP 6M",
        "KW SWAP 9M",
        "KW SWAP 1Y",
        "KW SWAP 2Y",
        "KW SWAP 3Y",
        "KW SWAP 4Y",
        "KW SWAP 5Y",
        "KW SWAP 7Y",
        "KW SWAP 10Y"
    ]

    rates = get_price_from_sql(date.today() - timedelta(days=10),
                               date.today(),
                               Name_rates,
                               type='w')[Name_rates]

    data = rates.iloc[-1]

#    ref = ql.Date.from_date(data.name)
    ref = ql.Date.from_date(date.today())

#    Tenor = ['1D', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']
    Tenor = ['3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']

    calendar = ql.TARGET()

    Maturity = [
        # ql.SouthKorea().advance(ref, ql.Period(1, ql.Days)),
        calendar.advance(ref, ql.Period(3, ql.Months)),
        calendar.advance(ref, ql.Period(6, ql.Months)),
        calendar.advance(ref, ql.Period(9, ql.Months)),
        calendar.advance(ref, ql.Period(1, ql.Years)),
        calendar.advance(ref, ql.Period(2, ql.Years)),
        calendar.advance(ref, ql.Period(3, ql.Years)),
        calendar.advance(ref, ql.Period(4, ql.Years)),
        calendar.advance(ref, ql.Period(5, ql.Years)),
        calendar.advance(ref, ql.Period(7, ql.Years)),
        calendar.advance(ref, ql.Period(10, ql.Years))
    ]

    Type = [
        # 'cash',
        'cash',
        'swap',
        'swap',
        'swap',
        'swap',
        'swap',
        'swap',
        'swap',
        'swap',
        'swap'
    ]

    df = pd.DataFrame({'Rates': list(data),
                       'Maturity': np.nan,
                       'DaysToMaturity': np.nan,
                       'Type': Type}, index=Tenor)

    for i, v in enumerate(df.index):
        df.loc[v, 'Maturity'] = ql.Date.to_date(Maturity[i])
        df.loc[v, 'DaysToMaturity'] = (df.loc[v, 'Maturity'] - data.name).days

    return df


def get_curve(today, quote):

    depo = quote[quote['Type'] == 'cash']
    swap = quote[quote['Type'] == 'swap']

    todays_date = ql.Date.from_date(today)
    ql.Settings.instance().evaluationDate = todays_date
    calendar = ql.TARGET()
    dayCounter = ql.Actual365Fixed()
    convention = ql.ModifiedFollowing
    settlementDays = 0
    frequency = ql.Quarterly

    depositHelpers = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate / 100)),
                                           ql.Period(int(day), ql.Days),
                                           settlementDays,
                                           calendar,
                                           convention,
                                           False, #endOfMonth
                                           dayCounter)
                      for day, rate in zip(depo['DaysToMaturity'], depo['Rates'])]

    swapHelpers = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate / 100)),
                                     ql.Period(int(day), ql.Days),
                                     calendar,
                                     frequency,
                                     convention,
                                     dayCounter,
                                     ql.EURLibor3M())
                   for day, rate in zip(swap['DaysToMaturity'], swap['Rates'])]

    helpers = depositHelpers + swapHelpers

    curve = ql.PiecewiseLinearZero(todays_date, helpers, dayCounter)
    curve_ff = ql.PiecewiseFlatForward(todays_date, helpers, dayCounter)
    return curve


def discount_factor(dt, curve):
    dt = ql.Date.from_date(dt)

    return curve.discount(dt)


def zero_rate(dt, curve):
    dt = ql.Date.from_date(dt)
    day_counter = ql.Actual360()
    compounding = ql.Compounded
    freq = ql.Continuous
    zerorate = curve.zeroRate(dt,
                              day_counter,
                              compounding,
                              freq).rate()

    return zerorate


def forward_rate(dt, curve):
    dt = ql.Date.from_date(dt)
    day_counter = ql.Actual360()
    compounding = ql.Compounded
    freq = ql.Continuous
    forwardrate = curve.forwardRate(dt,
                                    dt,
                                    day_counter,
                                    compounding,
                                    freq,
                                    True).rate()

    return forwardrate


def get_quote():
    quote = get_KRWIRSdata()
    curve = get_curve(date.today(), quote)

    quote['discount factor'] = np.nan
    quote['zero rate'] = np.nan
    quote['forward rate'] = np.nan

    for tenor, dt in zip(quote.index, quote['Maturity']):
        quote.loc[tenor, 'discount factor'] = discount_factor(dt, curve)
        quote.loc[tenor, 'zero rate'] = zero_rate(dt, curve) * 100
        quote.loc[tenor, 'forward rate'] = forward_rate(dt, curve) * 100

    return quote


if __name__ == "__main__":
    df = get_quote()
    print(df)

