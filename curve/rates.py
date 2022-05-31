import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By


class SeleniumCrawler:
    chromedriver = r"C:\selenium\chromedriver.exe"
    IRS_url = "http://www.smbs.biz/Exchange/IRS.jsp"
    CD_url = 'http://www.smbs.biz/Bond/BondMajor.jsp'
    Treasury_url = 'http://www.smbs.biz/Bond/BondMajor.jsp'

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('headless')
        self.options.add_argument('window-size=1920x1080')
        self.options.add_argument("disable-gpu")

    def IRSrate(self):

        driver = webdriver.Chrome(self.chromedriver, options=self.options)
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
        driver = webdriver.Chrome(self.chromedriver, options=self.options)
        driver.get(self.CD_url)

        data_CD91 = driver.find_elements(by=By.XPATH,
                                         value="//*[@id='objContents2']/div[5]/table/tbody/tr")[10]

        CD91 = data_CD91.text.split()[1]

        df = pd.DataFrame({'rate': [CD91]}, index=['91D'])

        return df

    def Treasuryrate(self):
        driver = webdriver.Chrome(self.chromedriver, options=self.options)
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
    irs = SeleniumCrawler()
    print(irs.IRSrate())
    print(irs.CDrate())
    print(irs.Treasuryrate())
