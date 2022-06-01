import pandas as pd
from datetime import date

from bs4 import BeautifulSoup
import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

"""
금리 곡선을 도출하기 위한 데이터 수집
단기(1년미만) : 콜금리 + CD 3개월
장기(1년이상) : Swap rate
"""


def RatesCrawlerBS4(date: date) -> pd.DataFrame:
    """ 주말이면 전영업일로 조회됨 """
    url = 'https://www.bondweb.co.kr/Prime_web/menu03/sub08/kdb/swap_01LP2.asp?udate='

    res = requests.get(url + date.strftime("%Y-%m-%d"))
    data = BeautifulSoup(res.text, 'html.parser')

    data = data.select('body>table')[1].select('tbody > tr')

    maturity_list = []
    rate_list = []
    for dat in data:
        info = dat.get_text().split()
        maturity = f'{info[0]}Y'
        mid_rate = (float(info[2]) + float(info[4])) / 2

        maturity_list.append(maturity)
        rate_list.append(mid_rate)

    df = pd.DataFrame({'rate': rate_list}, index=maturity_list)

    return df


class RatesCrawlerSelenium:
    chromedriver = r"C:\Users\tjdal\OneDrive\문서\.wdm\drivers\chromedriver\win32\102.0.5005.61\chromedriver.exe"
    IRS_url = "http://www.smbs.biz/Exchange/IRS.jsp"
    CD_url = 'http://www.smbs.biz/Bond/BondMajor.jsp'
    Treasury_url = 'http://www.smbs.biz/Bond/BondMajor.jsp'
    IRSall_url = 'https://www.bondweb.co.kr/Prime_web/menu03/sub08/kdb/swap_01LP2.asp?udate=2022-05-30'

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('headless')
        self.options.add_argument('window-size=1920x1080')
        self.options.add_argument("disable-gpu")

    def IRSrate(self):

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        driver.get(self.IRS_url)

        data_maturity = driver.find_element(by=By.XPATH,
                                            value="//*[@id='frm_SearchDate']/div[9]/table/thead/tr")
        maturity = data_maturity.text.split()[1:]

        data = driver.find_elements(by=By.XPATH,
                                    value="//*[@id='frm_SearchDate']/div[9]/table/tbody/tr")
        receive = data[0].text.split()[1:]
        mid = data[1].text.split()[1:]
        pay = data[2].text.split()[1:]

        df = pd.DataFrame({'rate': mid}, index=maturity)

        driver.quit()

        return df

    def CDrate(self):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        driver.get(self.CD_url)

        data_CD91 = driver.find_elements(by=By.XPATH,
                                         value="//*[@id='objContents2']/div[5]/table/tbody/tr")[10]

        CD91 = data_CD91.text.split()[1]

        df = pd.DataFrame({'rate': [CD91]}, index=['91D'])

        return df

    def Treasuryrate(self):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        driver.get(self.Treasury_url)

        data_T_3Y = driver.find_elements(by=By.XPATH,
                                         value="//*[@id='objContents2']/div[5]/table/tbody/tr")[0]

        data_T_5Y = driver.find_elements(by=By.XPATH,
                                         value="//*[@id='objContents2']/div[5]/table/tbody/tr")[1]

        data_T_10Y = driver.find_elements(by=By.XPATH,
                                          value="//*[@id='objContents2']/div[5]/table/tbody/tr")[2]

        data_T_20Y = driver.find_elements(by=By.XPATH,
                                          value="//*[@id='objContents2']/div[5]/table/tbody/tr")[3]

        T_3Y = data_T_3Y.text.split()[1]
        T_5Y = data_T_5Y.text.split()[1]
        T_10Y = data_T_10Y.text.split()[1]
        T_20Y = data_T_20Y.text.split()[1]

        df = pd.DataFrame({'rate': [T_3Y, T_5Y, T_10Y, T_20Y]}, index=['3Y', '5Y', '10Y', '20Y'])

        return df


if __name__ == "__main__":
    rates = RatesCrawlerSelenium()
    print(rates.IRSrate())
    # print(rates.CDrate())
    # print(rates.Treasuryrate())

    ref = date.today()
    print(RatesCrawlerBS4(ref))
