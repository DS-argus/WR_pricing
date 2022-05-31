import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def GBMRandomGenerator(sName: list,
                       steps: int,
                       rf: float,
                       sigma: dict,
                       corr: np.array = None,
                       fixed_seed: bool = False) -> np.array:
    """
    rf, vol을 고려한 난수를 생성하여 dict로 return
    :param sName: list of names of Underlyings
    :param steps: days for simulation  ex) if m = 1,000, generates 1,000 random numbers
    :param rf: risk-free rate
    :param sigma: annual volatility of Underlyings
    :param fixed_seed: fixed randomness --> 고정시키면 underlying끼리 다르게 고정됨 Good
    :return: random number of Underlyings considering mean daily return and daily volatility
    """

    if fixed_seed:
        np.random.seed(1000)

    # 파라미터 추정 방법에 따라 다르게 설정 maybe...
    #dt = 1/365
    dt = 1/252

    # 난수 생성
    random_normal = np.random.standard_normal((len(sName), steps))

    # 상관관계 고려한 난수 생성
    if corr is None:
        corr = np.identity(n=len(sName))

    cholesky = np.linalg.cholesky(corr)

    random_correlated = np.matmul(cholesky, random_normal)

    # GBM 난수 생성
    random_GBM = np.zeros((len(sName), steps))
    for i, v in enumerate(sName):
        random_GBM[i] = (rf - 0.5 * sigma[v] ** 2) * dt + sigma[v] * np.sqrt(dt) * random_correlated[i]
        #random_GBM[i] = rf * dt + sigma[v] * np.sqrt(dt) * random_correlated[i]

    return random_GBM


def MatrixGenerator(path_matrix: np.array,
                    s_val: list = None,
                    chart: bool = False) -> pd.DataFrame:
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

    # same results
    # price_matrix = np.concatenate((s_val, s_val * np.exp(np.cumsum(path_matrix, axis=1))), axis=1)
    price_matrix = np.concatenate((s_val, s_val * np.cumproduct(np.exp(path_matrix), axis=1)), axis=1)

    if chart:
        length = len(price_matrix[0])
        x = np.linspace(0, length, length)
        for i in range(len(price_matrix)):
            plt.plot(x, price_matrix[i])

        plt.show()

    return pd.DataFrame(price_matrix).T


def GBMPathGenerator(sName: list,
                     steps: int,
                     rf: float,
                     sigma: dict,
                     corr: np.array = None,
                     fixed_seed: bool = False,
                     s_val: list = None,
                     chart: bool = False) -> pd.DataFrame:

    GBMrandom = GBMRandomGenerator(sName, steps, rf, sigma, corr, fixed_seed)

    price = MatrixGenerator(GBMrandom, s_val, chart)

    return price


if __name__ == "__main__":

    sName = ['KOSPI200']

    steps = 252

    rf = 0.03

    vol = {
        'KOSPI200': 0.2
    }

    price = GBMPathGenerator(sName, steps, rf, vol)




    ## check normality --> OK
    # iteration = 5000
    # last_price = np.zeros(iteration)
    #
    # for i in range(iteration):
    #
    #     price = GBMPathGenerator(sName, steps, rf, vol)
    #     last_price[i] = price.iloc[-1]
    #
    # rtn = np.log(last_price)
    # print(stats.normaltest(rtn))
    # print(stats.shapiro(rtn))
    # print(stats.skew(rtn))
    # print(stats.kurtosis(rtn))
    # print(stats.probplot(rtn, plot=plt))
    #
    # plt.show()



