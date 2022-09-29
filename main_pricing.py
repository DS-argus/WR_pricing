import time

import els.class_els
from els.class_els import *
from process.GBMQL import *
from pricing.parameters import *
from idxdata.historical_data import *
from curve.KRWIRScurve import get_KRWIRSdata, get_curve, discount_factor
import numpy as np
from datetime import date


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
            result[i] = round(vol, 6)

        return result

    def EWMA_vol(self, days: int, to_: date = date.today(), alpha: float = 0.94):

        result = dict()

        for i in self.underlying:
            vol = EWMA_vol([i], days, to_, alpha)
            result[i] = round(vol, 6)

        return result

    def implied_vol(self, period: str, dt: date = date.today()):

        result = dict()

        for i in self.underlying:
            if i != 'CSI300':
                vol = implied_vol([i], period, dt)
                result[i] = round(vol, 6)
            else:
                # CSI300은 LIVE IVOL 제공이 안되어 HVOL로 산출
                if period == '30d':
                    days = 30
                elif period == '60d':
                    days = 60
                elif period == '3m':
                    days = 90
                elif period == '6m':
                    days = 180

                vol = historical_vol([i], days, dt)
                result[i] = round(vol, 6)

        return result

    def historical_corr(self, days: int, to_: date = date.today()):

        result = historical_corr(self.underlying, days, to_)

        return result.to_numpy()

    def EWMA_corr(self, days: int, to_: date = date.today(), alpha: float = 0.94):

        result = EWMA_corr(self.underlying, days, to_, alpha)

        return result.to_numpy()

    def GBMprocess(self, spot: list = None) -> pd.DataFrame:
        maturity = self.els.maturity
        start_date = ql.Date.from_date(self.eval_date)
        calendar = self.els.get_calendar()
        underlying = self.underlying
        rf = self.rf
        sigma = self.sigma
        corr = self.corr.tolist()

        df_path = GBMprocess(maturity,
                             start_date,
                             calendar,
                             underlying,
                             rf,
                             sigma,
                             corr,
                             spot)

        return df_path

    def get_curve(self):
        curve_data = get_KRWIRSdata()
        curve = get_curve(self.eval_date, curve_data)
        return curve

    def get_price(self, simulation_num):

        # 할인율 커브생성
        curve = self.get_curve()

        pv_list = np.zeros(simulation_num)

        for i in range(simulation_num):

            self.els.df = self.GBMprocess()

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
    underlying = ['S&P500', 'EUROSTOXX50', 'CSI300']
    trading_date = date.today()
    maturity = 3  # 만기(단위:연)
    periods = 6  # 평가(단위:월)
    coupon = 0.087
    barrier = [0.90, 0.85, 0.80, 0.80, 0.75, 0.65]
    KI_barrier = 0.5

    # ELS 생성
    els1 = SimpleELS(underlying, trading_date, maturity, periods, coupon, barrier)
    els2 = KIELS(underlying, trading_date, maturity, periods, coupon, barrier, KI_barrier)

    # Pricing 실행
    epr = ELSPricing(els2)

    # process 생성을 위한 금리 --> 3Y zero rate from IRS curve
    rf = 0.0432

    # Historical, EWMA vol and corr
    hist_vol = epr.historical_vol(120)
    ewma_vol = epr.EWMA_vol(120)
    avg_vol = {i: (hist_vol.get(i, 0) + ewma_vol.get(i, 0)) / 2 for i in hist_vol.keys() | ewma_vol.keys()}

    avg_corr = (epr.historical_corr(120) + epr.EWMA_corr(120)) / 2

    # Implied vol
    ivol_30d = epr.implied_vol('30d')
    ivol_60d = epr.implied_vol('60d')
    ivol_3m = epr.implied_vol('3m')
    ivol_6m = epr.implied_vol('6m')


    # 시뮬레이션
    epr.rf = rf
    epr.sigma = ivol_30d
    epr.corr = avg_corr

    simulation_num = 5000

    start = time.time()
    print(epr.get_price(simulation_num))
    print(time.time()-start)


