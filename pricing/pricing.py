"""
1. elsclass에 pv 메서드 만들까? -> get_result에서 리턴하고 month나오니까 month로 해도 되고 day로 해도 되고 만들 순 있음
 --> 메서드로 만들면 고정 할인율 쓰는 식으로
 --> 만약 IRS 금리 곡선 뽑아내서 할거면 pv 구하는 함수 따로 만드는게 나을 듯

2. 만들면 GBM 프로세스 1개 생성 후 -> get_pv 를 수많이 반복해서 pricing

3. 이때 multiprocessing 적용 가능하게 할 수 있나?
 --> 시뮬레이션 횟수가 100000이면 적당히 4등분 해서 각각 pv 구한뒤 평균내는 식으로
 --> multi_pricing(I/4): I는 시뮬 횟수


"""
import els.class_els
from els.class_els import SimpleELS, MPELS
from process.GBM import *

import numpy as np
from datetime import date


class ELSPricing:
    def __init__(self, els: els.class_els):
        self.els = els
        # Set evaluation date as today(default)
        self.els.start_date = date.today()
        self.underlying = self.els.get_info()['underlying']
        self.s_num = len(self.underlying)
        # generating longer periods of forecasting than maturity of ELS
        self.period = (self.els.get_info()['maturity'] + 1) * 365

    def get_params(self, from_, to_):
        parameters = get_params(self.underlying, from_, to_)
        return parameters

    def run_simulation(self, simulation_num, discount_rate, r_info, v_info, corr=None):

        if corr is None:
            corr = np.identity(self.s_num)

        periods = (self.els.get_schedule()[-1] - self.els.start_date).days + 1

        pv_list = np.zeros(simulation_num)

        for i in range(simulation_num):
            process = GBM_path_for_pricing(self.s_num,
                                           self.underlying,
                                           periods,
                                           r_info,
                                           v_info,
                                           self.els.start_date,
                                           corr,
                                           fixed_seed=False,
                                           s_val=None,
                                           chart=False)

            self.els.df = process

            redemption_month = int(self.els.get_result()[0] / self.els.periods)
            redemption_date = self.els.get_schedule()[redemption_month - 1]
            days_to_redemption = (redemption_date - self.els.start_date).days

            els_return = self.els.get_result()[1]
            present_value = (1 + els_return) / (np.exp(-discount_rate * (days_to_redemption / 365)))

            pv_list[i] = present_value

        price = np.mean(pv_list)

        return price


if __name__ == "__main__":

    # ELS 정보(엑셀 시트 내에서 받아오는 걸로 수정해야함)
    underlying = ['NIKKEI225', 'EUROSTOXX50', 'CSI300']
    start_date = date.today()           # pricing class에 들어갈때는 상관없음. 초기화됨
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.0822
    barrier = [0.90, 0.90, 0.85, 0.80, 0.75, 0.60]

    # ELS 생성
    els1 = MPELS(underlying, start_date, maturity, periods, coupon, barrier,MP_barrier=0.6)

    # Pricing 실행
    epr = ELSPricing(els1)

    # 파라미터 생성
    r, v, corr = epr.get_params(date(2021, 1, 1), date(2022, 4, 22))

    # 시뮬레이션
    sim_num = 1000
    discount_rate = 0.01
    result = epr.run_simulation(sim_num, discount_rate, r, v, corr)

    print(result)
