import time

import els.class_els
from els.class_els import *
from process.GBMQL import *
from params.parameters import *
from idxdata.historical_data import *
from curve.KRWIRScurve import get_KRWIRSdata, get_curve, discount_factor, get_quote
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

        return price

def print_to_excel():

    wb = xw.Book.caller()
    ws = wb.sheets['main']

    # get inputs
    start_date = date.today()
    underlying = [i for i in ws.range("B4:D4").value if i is not None]
    maturity = int(ws.range("B5").value)
    periods = int(ws.range("B6").value)
    coupon = ws.range("B7").value
    barrier = [float(i) / 100 for i in ws.range("B8").value.split("-")]
    KI_barrier = ws.range("B9").value
    MP_barrier = ws.range("B10").value

    Lizard_barrier = {ws.range("B11").value: ws.range("B12").value,
                      ws.range("C11").value: ws.range("C12").value,
                      ws.range("D11").value: ws.range("D12").value}
    Lizard_barrier = {int(k): float(v) / 100 for k, v in Lizard_barrier.items() if v is not None}

    # create ELS class
    if ws.range("B3").value == "일반 ELS":
        els = SimpleELS(underlying, start_date, maturity, periods, coupon, barrier)
    elif ws.range("B3").value == "낙인 ELS":
        els = KIELS(underlying, start_date, maturity, periods, coupon, barrier, KI_barrier)
    elif ws.range("B3").value == "리자드 ELS":
        els = LizardELS(underlying, start_date, maturity, periods, coupon, barrier, Lizard_barrier, 1)
    elif ws.range("B3").value == "리자드 낙인 ELS":
        els = LizardKIELS(underlying, start_date, maturity, periods, coupon, barrier, KI_barrier, Lizard_barrier, 1)
    elif ws.range("B3").value == "월지급 ELS":
        els = MPELS(underlying, start_date, maturity, periods, coupon, barrier, MP_barrier)

    # create ELS pricing class
    epr = ELSPricing(els)

    # get pricing parameters
    rf = get_quote().loc[str(maturity) + "Y", 'zero rate'] / 100
    simulation_num = int(ws.range("G5").value)

    if ws.range("G3").value == "Hvol":
        vol = epr.historical_vol(int(ws.range("H3").value))

    elif ws.range("G3").value == "EWMAvol":
        vol = epr.EWMA_vol(int(ws.range("H3").value))

    elif ws.range("G3").value == "AVGvol":
        Hvol = epr.historical_vol(int(ws.range("H3").value))
        EWMAvol = epr.EWMA_vol(int(ws.range("H3").value))
        vol = {i: round(((Hvol.get(i, 0) + EWMAvol.get(i, 0)) / 2), 4)
               for i in Hvol.keys() | EWMAvol.keys()}

    elif ws.range("G3").value == "Ivol":
        vol = epr.implied_vol(ws.range("H3").value)

    if ws.range("G4").value == "Hcorr":
        corr = epr.historical_corr(int(ws.range("H4").value))

    elif ws.range("G4").value == "EWMAcorr":
        corr = epr.EWMA_corr(int(ws.range("H4").value))

    elif ws.range("G4").value == "AVGcorr":
        Hcorr = epr.historical_corr(int(ws.range("H4").value))
        EWMAcorr = epr.EWMA_corr(int(ws.range("H4").value))
        corr = (Hcorr + EWMAcorr) / 2

    # setting pricing class
    epr.rf = rf
    epr.sigma = vol
    epr.corr = corr

    start = time.time()
    price = epr.get_price(simulation_num)
    cons_ts = time.time() - start

    ws.range("K4:L11").clear_contents()
    ws.range("N8:P11").clear_contents()

    ws.range("K3").value = price
    ws.range("K5").value = vol
    ws.range("N5").value = corr
    ws.range("K1").value = f"{cons_ts:.5f} sec"  # time
    ws.range("M1").value = time.strftime("%y/%m/%d %H:%M:%S", time.localtime())

    return


if __name__ == "__main__":

    xw.Book(r"\\172.31.1.222\Deriva\자동화\자동화폴더\구조화증권위험분석\구조화증권위험분석.xlsm").set_mock_caller()
    print_to_excel()

