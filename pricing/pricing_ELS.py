import numpy as np
import pandas as pd
from idxdata.historical_data import get_hist_data
from els.class_els import SimpleELS
from process.GBM import *
from datetime import date, timedelta

def get_PV(els, df_price, discount_rate): # 할인율, Sim 횟수
    els.df = df_price

    els_repayment_month = int(els.get_result()[0]/els.periods)
    els_repayment_date = els.get_schedule()[els_repayment_month - 1]
    days_to_repayment = (els_repayment_date - els.start_date).days

    els_return = float(els.get_result()[1][:-1])/100  #0.03
    discount_value = (1+els_return) / (np.exp(-discount_rate * (days_to_repayment/365)))

    return discount_value


if __name__ == "__main__":
    #ELS 정보 및 ELS 생성
    S = ['KOSPI200', 'S&P500', 'HSCEI']
    sim_start_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.04
    barrier = [0.95, 0.90, 0.80, 0.75, 0.70, 0.60]
    els1 = SimpleELS(S, sim_start_date, maturity, periods, coupon, barrier)

    # 주가 경로 생성 위한 인덱스 정보
    df = get_hist_data()

    start_date = date(2005, 1, 1)
    end_date = date(2022, 4, 22)

    index_info, corr_matrix = get_params(S, start_date, end_date, df)

    u_1 = index_info['KOSPI200'][0]
    sigma_1 = index_info['KOSPI200'][1]

    u_2 = index_info['S&P500'][0]
    sigma_2 = index_info['S&P500'][1]

    u_3 = index_info['HSCEI'][0]
    sigma_3 = index_info['HSCEI'][1]

    # S0 = [1, 1, 1]
    #
    # result = []
    # for i in range(10000):
    #     #주가 경로 생성
    #     path_1 = gen_path(2000, u_1, sigma_1)
    #     path_2 = gen_path(2000, u_2, sigma_2)
    #     path_3 = gen_path(2000, u_3, sigma_3)
    #
    #     correlation = corr_matrix
    #
    #     path_matrix = gen_path_matrix(correlation, path_1=path_1, path_2=path_2, path_3=path_3)
    #
    #     price_matrix = gen_price_matrix(S0, path_matrix, chart=False)
    #
    #     df_price = cvt_to_df(S, price_matrix, sim_start_date)
    #
    #     result.append(get_PV(els1, df_price, 0.01))
    #
    #
    # result = np.array(result)
    #
    # print(np.mean(result))


