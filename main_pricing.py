import time

import els.class_els
from els.class_els import *
from process.GBM import *
from pricing.parameters import *
from idxdata.historical_data import *
from curve.KRWIRScurve import get_KRWIRSdata, get_curve, discount_factor
import numpy as np
from datetime import date

from multiprocessing import Pool, Process



class ELSPricing:
    def __init__(self, els: els.class_els, eval_date: date = date.today()):
        self.els = els
        self.eval_date = eval_date      # 무조건 >= today
        self.underlying = self.els.get_info()['underlying']
        self.s_num = len(self.underlying)

        self.rf = None
        self.sigma = None
        self.corr = None

        if self.els.start_date < date.today():
            self.past_price = get_price_from_sql(self.els.start_date,
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

    def get_curve(self):
        curve_data = get_KRWIRSdata()
        curve = get_curve(self.eval_date, curve_data)
        return curve

    def get_pv(self):
        process = self.GBMprocess(self.rf, self.sigma, self.corr)
        self.els.df = process
        curve = self.get_curve()

        redemption_month = int(self.els.get_result()[0])

        if isinstance(els, MPELS):
            redemption_date = self.els.get_schedule()[redemption_month - 1]
        else:
            idx = int(redemption_month / self.els.periods)
            redemption_date = self.els.get_schedule()[idx - 1]

        els_return = self.els.get_result()[1]
        # return 1 + els_return, redemption_date

        DF = discount_factor(redemption_date, curve)
        present_value = (1 + els_return) * DF

        return present_value

    def simulation(self, simulation_num):

        with Pool(processes=4) as pool:
            results = pool.starmap(self.get_pv, [() for _ in range(simulation_num)])

        return round(np.mean(np.array(results)) * 100, 2)

    def get_price(self, simulation_num):

        rf = self.rf
        sigma = self.sigma
        corr = self.corr

        # 할인율 커브생성
        curve = self.get_curve()

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

            els_return = self.els.get_result()[1]

            DF = discount_factor(redemption_date, curve)
            present_value = (1 + els_return) * DF

            pv_list[i] = present_value

        price = np.mean(pv_list)

        return round(price * 100, 2)


if __name__ == "__main__":

    # ELS 정보
    underlying = ['S&P500', 'EUROSTOXX50']
    #trading_date = date(2022, 1, 1)
    trading_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.02
    barrier = [0.95, 0.85, 0.80, 0.80, 0.75, 0.70]
    KI_barrier = 0.5

    # ELS 생성
    els1 = SimpleELS(underlying, trading_date, maturity, periods, coupon, barrier)
    els2 = KIELS(underlying, trading_date, maturity, periods, coupon, barrier, KI_barrier)

    # Pricing 실행
    epr = ELSPricing(els2)

    # process 생성을 위한 금리 --> 3Y zero rate from IRS curve
    rf = 0.0367

    # vol, corr for 6 months
    hist_vol = epr.historical_vol(120)
    ewma_vol = epr.EWMA_vol(120)
    sigma = {i: (hist_vol.get(i, 0) + ewma_vol.get(i, 0)) / 2 for i in hist_vol.keys() | ewma_vol.keys()}

    corr = (epr.historical_corr(120) + epr.EWMA_corr(120)) / 2

    print(sigma)
    print(corr)

    # 시뮬레이션
    epr.rf = rf
    epr.sigma = sigma
    epr.corr = corr

    simulation_num = 5000

    # start = time.time()
    # print(epr.simulation(simulation_num))
    # print(time.time()-start)

    start = time.time()
    print(epr.get_price(simulation_num))
    print(time.time()-start)


