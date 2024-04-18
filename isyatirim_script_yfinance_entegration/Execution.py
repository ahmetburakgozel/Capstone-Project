import os
from datetime import datetime

import pandas as pd
from isyatirim_script_yfinance_entegration import HistoricalData as sp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


first_df = pd.read_excel('isyatirim_script_yfinance_entegration/Endeks.xlsx')
first_df["BIST 100"].dropna(inplace=True)

if not os.path.exists('stock_dfs'):
    os.makedirs('stock_dfs')

start_date = datetime(2014, 4, 17)
end_date = datetime(2024, 4, 17)

for ticker in first_df["BIST 100"]:
    if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        df = sp.stock_prices(ticker+".IS", start_date, end_date)
        df.to_csv('stock_dfs/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))

