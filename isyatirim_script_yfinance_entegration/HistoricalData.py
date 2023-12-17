import pandas as pd

from pandas_datareader import data as pdr

import yfinance


def stock_prices(symbols, start_date, end_date):
    yfinance.pdr_override()
    df = pdr.get_data_yahoo(symbols, start_date, end_date)
    print(df.columns.tolist())

    df.dropna(inplace=True)

    print(df.tail())
    print(df.shape)
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
