import els.class_els
from els.class_els import *
from process.GBM import *
from parameters import *

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

    def historical_vol(self, days: int, to_: date = date.today()):

        result = dict()

        for i in self.underlying:
            vol = historical_vol([i], days, to_)
            result[i] = vol

        return result

    def EWMA_vol(self, days: int, to_: date = date.today(), alpha: float = 0.94):

        result = dict()

        for i in self.underlying:
            vol = EWMA_vol([i], days, to_, alpha)
            result[i] = vol

        return result

    def historical_corr(self, days: int, to_: date = date.today()):

        result = historical_corr(self.underlying, days, to_)

        return result.to_numpy()

    def EWMA_corr(self, days: int, to_: date = date.today(), alpha: float = 0.94):

        result = EWMA_corr(self.underlying, days, to_, alpha)

        return result.to_numpy()

    def GBMprocess(self, rf, sigma, corr=None):

        if corr is None:
            corr = np.identity(self.s_num)

        if self.els.start_date < date.today():
            last_date = self.past_price.index[-1]
            s_val = self.past_price.loc[last_date, self.underlying]
            steps = (self.els.get_schedule()[-1] - date.today()).days

            process = GBMPathGenerator(self.underlying,
                                       steps,
                                       rf,
                                       sigma,
                                       corr=corr,
                                       fixed_seed=False,
                                       s_val=s_val,
                                       chart=False)
            process.index = pd.date_range(date.today(), self.els.get_schedule()[-1]).date
            process.columns = self.underlying
            GBMprocess = pd.concat([self.past_price, process])

        else:
            steps = (self.els.get_schedule()[-1] - date.today()).days

            process = GBMPathGenerator(self.underlying,
                                       steps,
                                       rf,
                                       sigma,
                                       corr=corr,
                                       fixed_seed=False,
                                       chart=False)

            process.index = pd.date_range(date.today(), self.els.get_schedule()[-1]).date
            process.columns = self.underlying

            GBMprocess = process

        return GBMprocess

    def get_price(self, simulation_num, discount_rate, rf, sigma, corr=None):

        pv_list = np.zeros(simulation_num)

        for i in range(simulation_num):

            process = self.GBMprocess(rf, sigma, corr)
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
    underlying = ['S&P500', 'EUROSTOXX50', 'KOSPI200']
    #trading_date = date(2022, 1, 1)
    trading_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.061
    barrier = [0.75, 0.75, 0.75, 0.75, 0.75, 0.65]

    # ELS 생성
    els1 = SimpleELS(underlying, trading_date, maturity, periods, coupon, barrier)

    # Pricing 실행
    epr = ELSPricing(els1)

    # get_param 안쓰고 직접 입력할 때 아래 두 줄 주석 해제하고 직접 입력(단위: 일)
    rf = 0.03
    v = epr.EWMA_vol(180)
    corr = epr.historical_corr(180)
    print(v)
    print(corr)

    v = {'S&P500': 0.2993,
         'EUROSTOXX50': 0.3096,
         'KOSPI200': 0.2517}

    corr = [
        [1, 0.4062, 0.3013],
        [0.4062, 1, 0.3905],
        [0.3013, 0.3905, 1]
    ]

    df = epr.GBMprocess(rf, v, corr)

    # 시뮬레이션
    simulation = 10000
    discount_rate = 0.03

    print(epr.get_price(simulation, discount_rate, rf, v, corr) * 10000)
