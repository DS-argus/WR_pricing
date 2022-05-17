"""
1. elsclass에 pv 메서드 만들까? -> get_result에서 리턴하고 month나오니까 month로 해도 되고 day로 해도 되고 만들 순 있음
 --> 메서드로 만들면 고정 할인율 쓰는 식으로
 --> 만약 IRS 금리 곡선 뽑아내서 할거면 pv 구하는 함수 따로 만드는게 나을 듯

2. 만들면 GBM 프로세스 1개 생성 후 -> get_pv 를 수많이 반복해서 pricing

3. 이때 multiprocessing 적용 가능하게 할 수 있나?
 --> 시뮬레이션 횟수가 100000이면 적당히 4등분 해서 각각 pv 구한뒤 평균내는 식으로
 --> multi_pricing(I/4): I는 시뮬 횟수


"""
from els.class_els import SimpleELS
from idxdata.historical_data import get_hist_data
from process.GBM import *

import numpy as np
from datetime import date
import multiprocessing
import multiprocess


def get_PV(els, df_price, discount_rate):  # 할인율, Sim 횟수

    els.df = df_price
    els.start_date = df_price.index[0]
    # return els.get_schedule()
    els_repayment_month = int(els.get_result()[0]/els.periods)
    els_repayment_date = els.get_schedule()[els_repayment_month - 1]
    days_to_repayment = (els_repayment_date - els.start_date).days

    els_return = els.get_result()[1]  # 0.03
    # discount_value = 1 + els_return
    discount_value = (1+els_return) / (np.exp(-discount_rate * (days_to_repayment/360)))

    return discount_value

def process_for_multi_sim():
    


if __name__ == "__main__":

    # ELS 정보(엑셀 시트 내에서 받아오는 걸로 수정해야함)
    underlying = ['KOSPI200', 'EUROSTOXX50']  # 기초자산
    start_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.0
    barrier = [0.95, 0.90, 0.80, 0.75, 0.70, 0.60]

    df = get_hist_data()

    # ELS 생성
    els1 = SimpleELS(underlying, start_date, maturity, periods, coupon, barrier, df)

    # 파라미터 측정 기간
    param_start_date = date(2021, 1, 1)
    param_end_date = date(2022, 4, 22)

    return_info, vol_info, corr_matrix = get_params(underlying, param_start_date, param_end_date, df)

    s_num = len(underlying)
    period = 365 * 4

    sim = 10000

    def multi_sim(path, i):

    pv_arr = np.zeros(sim)


    for i in range(sim):
        s_path = GBM_path_for_pricing(s_num, underlying, period, return_info, vol_info,
                                      start_date=start_date, corr=corr_matrix, chart=False)

        pv_arr[i] = get_PV(els1, s_path, 0.02)

    pool = multiprocessing.Pool(4)

    result = pool.starmap(func_for_multi, list_for_multi)

    result_1 = result[0]
    result_2 = result[1]
    result_3 = result[2]
    result_4 = result[3]

    pool.close()
    pool.join()

    fin_result = (result_1 + result_2 + result_3 + result_4) / 4

    print(f'price of ELS = {fin_result:.4f}')

