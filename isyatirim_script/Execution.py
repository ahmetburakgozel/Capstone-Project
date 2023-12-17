import os
import pandas as pd
from isyatirim_script import HistoricalData as sp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

first_df = pd.read_excel("isyatirim_script/Endeks.xlsx")
first_df["BIST 100"].dropna(inplace=True)

if not os.path.exists('isyatirim_script_stock_dfs'):
    os.makedirs('isyatirim_script_stock_dfs')

sdate = "15-12-2013"
edate = "15-12-2023"

for ticker in first_df["BIST 100"]:
    if not os.path.exists('isyatirim_script_stock_dfs/{}.csv'.format(ticker)):
        df = sp.stock_prices(ticker, sdate, edate)
        df.to_csv('isyatirim_script_stock_dfs/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))
