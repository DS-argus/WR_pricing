import els.class_els
from els.class_els import SimpleELS, MPELS
from process.GBM import *

import numpy as np
from datetime import date


class ELSPricing:
    def __init__(self, els: els.class_els, eval_date: date = date.today()):
        self.els = els
        self.eval_date = eval_date      # 무조건 >= today
        if self.els.start_date < date.today():
            print('최초기준가를 입력해주세요') # 미래 주가 생성 후 비교해야함
            print('최근 종가를 입력해주세요') # 남은 날까지 전일 종가기준으로 주가 경로 생성
        self.underlying = self.els.get_info()['underlying']
        self.s_num = len(self.underlying)

    def get_params(self, from_, to_):
        parameters = get_params(self.underlying, from_, to_)
        return parameters

    def get_price(self, simulation_num, eval_date, discount_rate, r_info, v_info, corr=None):

        if corr is None:
            corr = np.identity(self.s_num)

        process_length = (self.els.get_schedule()[-1] - self.els.start_date).days + 1

        pv_list = np.zeros(simulation_num)

        for i in range(simulation_num):
            process = GBM_path_for_pricing(self.s_num,
                                           self.underlying,
                                           process_length,
                                           r_info,
                                           v_info,
                                           self.eval_date,
                                           corr,
                                           fixed_seed=False,
                                           s_val=None,
                                           chart=False)

            self.els.df = process

            redemption_month = int(self.els.get_result()[0] / self.els.periods)
            redemption_date = self.els.get_schedule()[redemption_month - 1]
            days_to_redemption = (redemption_date - self.eval_date).days

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
    els1 = MPELS(underlying, start_date, maturity, periods, coupon, barrier, MP_barrier=0.6)

    # Pricing 실행
    epr = ELSPricing(els1)

    # 파라미터 생성
    r, v, corr = epr.get_params(date(2021, 1, 1), date(2022, 4, 22))

    # 시뮬레이션
    sim_num = 1000
    discount_rate = 0.01
    result = epr.get_price(sim_num, discount_rate, r, v, corr)

    print(result)
