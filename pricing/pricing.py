import els.class_els
from els.class_els import *
from process.GBM import *

import numpy as np
from datetime import date


class ELSPricing:
    def __init__(self, els: els.class_els, eval_date: date = date.today()):
        self.els = els
        self.eval_date = eval_date      # 무조건 >= today
        self.underlying = self.els.get_info()['underlying']
        self.s_num = len(self.underlying)

        if self.els.start_date < date.today():
            self.past_price = get_hist_data_from_sql(self.els.start_date,
                                                     date.today(),
                                                     self.underlying,
                                                     type="w")

    def get_params(self, from_, to_):
        parameters = get_params(self.underlying, from_, to_)
        return parameters

    def GBMprocess(self, r_info, v_info, corr=None):

        if corr is None:
            corr = np.identity(self.s_num)

        process_length = (self.els.get_schedule()[-1] - self.els.start_date).days + 1

        if self.els.start_date < date.today():
            last_date = self.past_price.index[-1]
            s_val = self.past_price.loc[last_date, self.underlying]

            process = GBM_path_for_pricing(self.s_num,
                                           self.underlying,
                                           process_length,
                                           r_info,
                                           v_info,
                                           date.today(),
                                           corr,
                                           fixed_seed=False,
                                           s_val=s_val,
                                           chart=False)

            GBMprocess = pd.concat([self.past_price, process])

        else:
            process = GBM_path_for_pricing(self.s_num,
                                           self.underlying,
                                           process_length,
                                           r_info,
                                           v_info,
                                           date.today(),
                                           corr,
                                           fixed_seed=False,
                                           s_val=None,
                                           chart=False)


            GBMprocess = process

        return GBMprocess

    def get_price(self, simulation_num, discount_rate, r_info, v_info, corr=None):

        pv_list = np.zeros(simulation_num)

        for i in range(simulation_num):

            process = self.GBMprocess(r_info, v_info, corr)
            self.els.df = process

            redemption_month = int(self.els.get_result()[0])

            if isinstance(self.els, MPELS):
                redemption_date = self.els.get_schedule()[redemption_month - 1]
            else:
                idx = int(redemption_month / self.els.periods)
                redemption_date = self.els.get_schedule()[idx - 1]

            if redemption_date < self.eval_date:
                continue

            days_to_redemption = (redemption_date - self.eval_date).days

            els_return = self.els.get_result()[1]
            discount_factor = np.exp(-discount_rate * (days_to_redemption / 365))
            present_value = (1 + els_return) * discount_factor

            pv_list[i] = present_value

        price = np.mean(pv_list)

        return price


if __name__ == "__main__":

    # ELS 정보
    underlying = ['S&P500', 'EUROSTOXX50', 'CSI300']
    trading_date = date(2022, 1, 25)
    #trading_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.0732
    barrier = [0.90, 0.90, 0.85, 0.80, 0.75, 0.60]

    # ELS 생성
    els1 = MPELS(underlying, trading_date, maturity, periods, coupon, barrier, MP_barrier=0.6)
    #els1 = SimpleELS(underlying, trading_date, maturity, periods, coupon, barrier)
    # Pricing 실행
    epr = ELSPricing(els1)

    # 파라미터 생성
    r, v, corr = epr.get_params(date(2021, 11, 23), date.today())

    # # get_param 안쓰고 직접 입력할 때 아래 두 줄 주석 해제하고 직접 입력(단위: 일)
    # r = {'KOSPI200': 0.0001119, 'S&P500': 0.0001119, 'HSCEI': 0.0001119}
    # v = {'KOSPI200': 0.001, 'S&P500': 0.001, 'HSCEI': 0.001}


    # 시뮬레이션
    sim_num = 1000
    discount_rate = 0.01

    result = epr.get_price(sim_num, discount_rate, r, v, corr)

    print(result)
