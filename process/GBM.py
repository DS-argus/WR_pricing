from idxdata.historical_data import get_hist_data_from_sql

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date, timedelta


def get_params(u_name: list, start: date, end: date) -> (dict, dict, np.array):
    """
    기초자산 종가 dataframe 받아서 지정한 구간의 daily 평균수익률, daily 변동성, corr matrix return
    :param u_name: names of indexes
    :param start: start date for parameter calculation
    :param end: end date for parameter calculation
    :return: return_dict, vol_dict, corr_dataframe
    """

    daily_return = dict()
    daily_vol = dict()

    # get return and volatility
    for underlying in u_name:
        data = np.array(get_hist_data_from_sql(start, end, [underlying])).astype(float)

        daily_log_return = np.log([data[1:] / data[:-1]])

        r = np.mean(daily_log_return)
        v = np.std(daily_log_return)

        daily_return[underlying] = r
        daily_vol[underlying] = v

    # get correlation matrix
    data_all = get_hist_data_from_sql(start, end, u_name)

    corr = data_all.corr(method='pearson').to_numpy()

    return daily_return, daily_vol, corr


def gen_path(u_num: int, m: int, u: dict, sigma: dict, fixed_seed: bool = False) -> dict:
    """
    return, vol을 고려한 난수를 생성하여 dict로 return
    :param u_num: number of Underlyings
    :param m: periods == number of days to simulate, ex) if m = 1,000, generates 1,000 random numbers
    :param u: mean daily return of Underlyings
    :param sigma: dialy volatility of Underlyings
    :param fixed_seed: fixed randomness --> 고정시키면 underlying끼리 다르게 고정됨 Good
    :return: random number of Underlyings considering mean daily return and daily volatility
    """

    if fixed_seed:
        np.random.seed(1000)

    u = np.array(list(u.values())).astype(float)
    sigma = np.array(list(sigma.values())).astype(float)

    path_dict = {}
    for i in range(u_num):
        random = np.random.standard_normal(m)
        random_with_daily_params = (u[i] - 0.5 * sigma[i] ** 2) + sigma[i] * random

        path_dict[i] = random_with_daily_params

    return path_dict


def gen_path_matrix(path: dict, corr: np.array = None) -> np.array:
    """
    gen_path 함수로 생성된 dict 형태의 path를 상관계수를 적용한 뒤 array로 return
    :param path: return from gen_path
    :param corr: correlation matrix
    :return: correlated path matrix that has shape of (u_num, periods)
    """

    if corr is None:
        corr = np.identity(n=len(path.keys()))

    if len(path.keys()) != len(corr):
        raise Exception("path의 수와 상관계수 행렬의 크기가 일치하지 않습니다")

    elif corr.size != len(path.keys()) ** 2:
        raise Exception("path의 수와 상관계수 행렬의 크기가 일치하지 않습니다")

    path_lengths = [len(path[i]) for i in path.keys()]

    if len(set(path_lengths)) != 1:
        raise Exception("path들의 구간 개수가 일치하지 않습니다")

    path_len = path_lengths[0]

    path_mat = np.arange(path_len)

    for index in path.keys():
        path_mat = np.vstack((path_mat, path[index]))

    path_mat = np.delete(path_mat, 0, axis=0)

    cholesky = np.linalg.cholesky(corr)

    path_mat = np.matmul(cholesky, path_mat)

    return path_mat


def gen_price_matrix(path_matrix: np.array, s_val: list = None, chart: bool = False) -> np.array:
    """
    gen_path_matrix 함수의 결과값을 활용해 GBM을 따르는 주가 경로 생성, 기초자산 초기값 및 차트는 옵션
    :param path_matrix: return from gen_path_matrix
    :param s_val: first value of each underlyings
    :param chart: show chart or not
    :return: matrix of price of underlyings that follows GBM
    """

    if s_val is None:
        s_val = [1] * len(path_matrix)

    if len(s_val) != len(path_matrix):
        raise Exception("S0의 개수와 path의 개수가 일치하지 않습니다")

    s_val = np.array(s_val).reshape((len(s_val), 1))

    path_matrix = np.array(path_matrix)

    price_matrix = np.concatenate((s_val, s_val * np.exp(np.cumsum(path_matrix, axis=1))), axis=1)

    if chart:
        length = len(price_matrix[0])
        x = np.linspace(0, length, length)
        for i in range(len(price_matrix)):
            plt.plot(x, price_matrix[i])

        plt.show()

    return price_matrix


def cvt_to_df(s:list, price_matrix: np.array, start: date = date.today()) -> pd.DataFrame:
    """
    주가 경로 matrix를 dataframe 형태로 변환 -> class_els랑 연결시키려고 dataframe 형태로 변환
    :param s: index for column name
    :param price_matrix: stock process matrix / (u_num X period+1) shape
    :param start: first day of future stock process
    :return: dataframe / (period+1, S) shape
    """

    df_price = pd.DataFrame(price_matrix).transpose()

    # 휴장, 영업일 고려할 경우 수정 필요
    index_date = [start + timedelta(days=i) for i in range(len(df_price.index))]

    df_price.index = index_date
    df_price.columns = s

    return df_price


def GBM_path_for_pricing(num_s: int,
                         s_name: list,
                         m: int,
                         u: dict,
                         sigma: dict,
                         start_date: date = date.today(),
                         corr: np.array = None,
                         fixed_seed: bool = False,
                         s_val: list = None,
                         chart: bool = False):

    """ get_params 함수를 제외한 나머지 함수들을 합쳐놓은 함수 """

    path_dict = gen_path(num_s, m, u, sigma, fixed_seed)

    path_mat = gen_path_matrix(path_dict, corr)

    price_mat = gen_price_matrix(path_mat, s_val, chart)

    price_df = cvt_to_df(s_name, price_mat, start_date)

    return price_df


if __name__ == "__main__":

    s_name = ['KOSPI200', 'S&P500', 'HSCEI']

    # 파라미터 측정 기간
    start_date = date(2021, 1, 1)
    end_date = date(2022, 4, 22)

    return_info, vol_info, corr_matrix = get_params(s_name, start_date, end_date)
    print(return_info)
    print(vol_info)

    # # get_param 안쓰고 직접 입력할 때 아래 두 줄 주석 해제하고 직접 입력(단위: 일)
    # return_info = {'KOSPI200': 0.001, 'S&P500': 0.005, 'HSCEI': 0.007}
    # vol_info = {'KOSPI200': 0.001, 'S&P500': 0.005, 'HSCEI': 0.008}



    s_num = len(s_name)
    period = 100000
    stock_path = GBM_path_for_pricing(s_num, s_name, period, return_info, vol_info, chart=False)

#    print(gen_path(s_num, 10000, return_info, vol_info, fixed_seed=True))
    dat = gen_path(s_num, 10000, return_info, vol_info, fixed_seed=True)
    print(return_info['KOSPI200']*252)
    print(np.mean(dat[0])*252)
    print("--------------------")
    print(return_info['S&P500']*252)
    print(np.mean(dat[1])*252)
    print("--------------------")
    print(return_info['HSCEI']*252)
    print(np.mean(dat[2])*252)
    print("--------------------")

    print("--------------------")
    print("--------------------")

    print(vol_info['KOSPI200']*np.sqrt(252))
    print(np.std(dat[0])*np.sqrt(252))
    print("--------------------")

    print(vol_info['S&P500']*np.sqrt(252))
    print(np.std(dat[1])*np.sqrt(252))

    print("--------------------")
    print(vol_info['HSCEI']*np.sqrt(252))
    print(np.std(dat[2])*np.sqrt(252))

    #
    # a = stock_path['KOSPI200']
    # ar = np.array(a).astype('float')
    # re = np.log([ar[1:] / ar[:-1]])
    # print(np.mean(re))
    #
    # a1 = stock_path['S&P500']
    # ar1 = np.array(a1).astype('float')
    # re1 = np.log([ar1[1:] / ar1[:-1]])
    # print(np.mean(re1))
    #
    # a2 = stock_path['HSCEI']
    # ar2 = np.array(a2).astype('float')
    # re2 = np.log([ar2[1:] / ar2[:-1]])
    # print(np.mean(re2))
