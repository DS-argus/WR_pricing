from idxdata.historical_data import get_price_from_sql

import numpy as np
import pandas as pd

from datetime import date, timedelta
from itertools import combinations


def historical_vol(name: list, days: int, to_: date = date.today()) -> float:

    data = get_price_from_sql(to_ - timedelta(days=days * 3),
                              to_,
                              name)

    data = data.iloc[-days-1:, :]

    data_array = data.to_numpy()

    log_return = np.log(data_array[1:] / data_array[:-1])
    variance = np.std(log_return, ddof=1) ** 2

    return np.sqrt(variance * 250)


def EWMA_vol(name: list, days: int, to_: date = date.today(), alpha: float = 0.94) -> float:

    data = get_price_from_sql(to_ - timedelta(days=days * 3),
                              to_,
                              name)

    data = data.iloc[-days-1:, :]
    data_array = data.to_numpy()

    # daily log return, length = days
    log_return = np.log(data_array[1:] / data_array[:-1])

    # (daily log return) ** 2
    log_return_squared = log_return ** 2

    # weight array
    length = len(log_return_squared)

    weight_array = (1-alpha) * (alpha ** np.arange(length)[::-1])
    weight = weight_array.reshape((length, 1))

    # EWMA variance
    EWMA_variance = np.sum(weight * log_return_squared)

    return np.sqrt(EWMA_variance * 250)


def implied_vol(name: list, period: str, to_: date = date.today()) -> float:

    if period == '30d':
        flds = '30DAY_IMPVOL_100.0%MNY_DF'
    elif period == '60d':
        flds = '60DAY_IMPVOL_100.0%MNY_DF'
    elif period == '3m':
        flds = '3MTH_IMPVOL_100.0%MNY_DF'
    elif period == '6m':
        flds = '6MTH_IMPVOL_100.0%MNY_DF'
    else:
        raise Exception('Check IVOL period again')

    data = get_price_from_sql(to_ - timedelta(days=5),
                              to_,
                              name,
                              flds,
                              ffill=False)

    return data.iloc[-1, 0] / 100


def historical_corr(name: list, days: int, to_: date = date.today()) -> pd.DataFrame:

    items = [i for i in range(len(name))]
    comb = list(combinations(items, 2))

    df_result = pd.DataFrame(index=name, columns=name)

    for i in comb:
        # select a single pair
        u1 = name[i[0]]
        u2 = name[i[1]]
        pair = [u1, u2]

        # get and modify data
        data = get_price_from_sql(to_ - timedelta(days=days * 3),
                                  to_,
                                  pair,
                                  ffill=False)

        data.dropna(axis=0, how="any", inplace=True)
        data = data.iloc[-days-1:, :]

        # daily log return
        data_array = data.to_numpy()
        log_return = np.log(data_array[1:] / data_array[:-1])

        # Historical correlation
        covariance = np.mean(log_return[:, 0] * log_return[:, 1])
        variance_1 = np.mean(log_return[:, 0] ** 2)
        variance_2 = np.mean(log_return[:, 1] ** 2)

        hist_corr = covariance / np.sqrt(variance_1 * variance_2)

        # fill dataframe
        df_result.loc[u1, u2] = hist_corr
        df_result.loc[u2, u1] = hist_corr

    df_result.fillna(value=1, inplace=True)

    return df_result

# ????
def EWMA_corr(name: list, days: int, to_: date = date.today(), alpha: float = 0.95) -> pd.DataFrame:

    items = [i for i in range(len(name))]
    comb = list(combinations(items, 2))

    df_result = pd.DataFrame(index=name, columns=name)

    for i in comb:
        # select a single pair
        u1 = name[i[0]]
        u2 = name[i[1]]
        pair = [u1, u2]

        # get and modify data
        data = get_price_from_sql(to_ - timedelta(days=days * 3),
                                  to_,
                                  pair,
                                  ffill=False)

        data.dropna(axis=0, how="any", inplace=True)
        data = data.iloc[-days-1:, :]

        # daily log return
        data_array = data.to_numpy()
        log_return = np.log(data_array[1:] / data_array[:-1])

        # weight array
        length = len(log_return)
        weight_array = (1 - alpha) * (alpha ** np.arange(length)[::-1])
        weight = weight_array.reshape((length, 1))

        # EWMA_covariance
        log_return_cross = (log_return[:, 0] * log_return[:, 1]).reshape((length, 1))
        EWMA_covariance = np.sum(weight * log_return_cross)

        log_return_squared_1 = log_return[:, 0] ** 2
        log_return_squared_2 = log_return[:, 1] ** 2

        EWMA_variance_1 = np.sum(weight * log_return_squared_1.reshape((length, 1)))
        EWMA_variance_2 = np.sum(weight * log_return_squared_2.reshape((length, 1)))

        # EWMA Correlation
        EWMA_corr = EWMA_covariance / (np.sqrt(EWMA_variance_1) * np.sqrt(EWMA_variance_2))

        # fill dataframe
        df_result.loc[u1, u2] = EWMA_corr
        df_result.loc[u2, u1] = EWMA_corr

    df_result.fillna(value=1, inplace=True)

    return df_result


if __name__ == "__main__":

    # print(f'{historical_vol(["S&P500"], 30) * 100:.3f}')
    # print(f'{historical_vol(["S&P500"], 60) * 100:.3f}')

    #print(f'{historical_vol(["S&P500"], 90) * 100:.3f}')

    dt = date.today()
    print(implied_vol(['KOSPI200'], '30d'))
    print(implied_vol(['EUROSTOXX50'], '30d'))
    print(implied_vol(['S&P500'], '30d'))
    print("----------------------------")

    print(EWMA_vol(['KOSPI200'], 30))
    print(EWMA_vol(['EUROSTOXX50'], 30))
    print(EWMA_vol(['S&P500'], 30))
    print("----------------------------")

    print(historical_vol(['KOSPI200'], 30))
    print(historical_vol(['EUROSTOXX50'], 30))
    print(historical_vol(['S&P500'], 30))
    print("----------------------------")



    # print(historical_corr(["S&P500", "EUROSTOXX50", "KOSPI200"], 90))
    # # print(historical_corr(["S&P500", "EUROSTOXX50", "HSCEI"], 90))


    # print(f'{historical_vol(["TESLA"], 120, date) * 100:.3f}')
    # print(f'{historical_vol(["TESLA"], 180, date) * 100:.3f}')
    # print(f'{historical_vol(["TESLA"], 360, date) * 100:.3f}')
