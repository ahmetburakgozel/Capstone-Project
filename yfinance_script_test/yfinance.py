from datetime import datetime

from pandas_datareader import data as pdr

import yfinance

import json
import urllib.request
import urllib.request

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

yfinance.pdr_override()

symbols = ['KCHOL.IS']
start_date = datetime(2013, 11, 13)
end_date = datetime(2023, 11, 13)
stock_data = pdr.get_data_yahoo(symbols, start_date, end_date)
print(stock_data.tail())

# DÜZELTİLECEK
# Adjusted close değeri, close değerinden çok uzak
stock_data.to_excel("yfinance_script_test/kchol.xlsx")


def stock_prices(stock, sdate, edate):
    sauce = urllib.request.urlopen(
        "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?hisse"
        "={}&startdate={}&enddate={}".format(stock, sdate, edate)).read()
    data = json.loads(sauce)
    df = pd.DataFrame(data["value"])
    print(df.columns.tolist())

    df.rename(columns={"HGDG_TARIH": "Date", "HGDG_KAPANIS": "Close",
                       "HGDG_MIN": "Min", "HGDG_MAX": "Max",
                       "HGDG_HACIM": "Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)

    print(df.tail())
    print(df.shape)
    return df[["Volume", "Close", "Min", "Max"]]
