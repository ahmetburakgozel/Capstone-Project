import time

# Yahoo Finance API
import yfinance as yf

import pandas as pd
import numpy as np
# Auto ARIMA
from pmdarima import auto_arima

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# bist100 hisse isimleri kısaltması
tickers = ['AEFES',
           'AGHOL',
           'AKBNK',
           'AKFGY',
           'AKSA',
           'AKSEN',
           'ALARK',
           'ALBRK',
           'ALGYO',
           'ALKIM',
           'ARCLK',
           'ASELS',
           'AYDEM',
           'BAGFS',
           'BERA',
           'BIMAS',
           'BRYAT',
           'BUCIM',
           'CEMTS',
           'CIMSA',
           'DEVA',
           'DOAS',
           'DOHOL',
           'ECILC',
           'EGEEN',
           'EKGYO',
           'ENJSA',
           'ENKAI',
           'ERBOS',
           'EREGL',
           'FROTO',
           'GARAN',
           'GENIL',
           'GESAN',
           'GLYHO',
           'GOZDE',
           'GSDHO',
           'GUBRF',
           'GWIND',
           'HALKB',
           'HEKTS',
           'IPEKE',
           'ISCTR',
           'ISDMR',
           'ISFIN',
           'ISGYO',
           'ISMEN',
           'JANTS',
           'KARSN',
           'KARTN',
           'KCHOL',
           'KONTR',
           'KORDS',
           'KOZAA',
           'KOZAL',
           'KRDMD',
           'LOGO',
           'MAVI',
           'MGROS',
           'NTHOL',
           'NUGYO',
           'ODAS',
           'OTKAR',
           'OYAKC',
           'PETKM',
           'PGSUS',
           'PRKAB',
           'PSGYO',
           'QUAGR',
           'SAHOL',
           'SASA',
           'SELEC',
           'SISE',
           'SKBNK',
           'SMRTG',
           'SNGYO',
           'SOKM',
           'TAVHL',
           'TCELL',
           'THYAO',
           'TKFEN',
           'TMSN',
           'TOASO',
           'TRGYO',
           'TSKB',
           'TSPOR',
           'TTKOM',
           'TTRAK',
           'TUKAS',
           'TUPRS',
           'TURSG',
           'ULKER',
           'VAKBN',
           'VESBE',
           'VESTL',
           'YATAS',
           'YKBNK',
           'YYLGD']

# tarih aralığı
start_date = '2013-11-13'
end_date = '2023-11-13'

# tahminler için boş bir DataFrame
# tarih, hisse senedi, düzeltilmiş kapanış fiyatı ve tahmin sütunları içerir
pred_df = pd.DataFrame(columns=['Date', 'Ticker', 'Adj Close', 'Prediction'])

# hisse senetlerini indir ve tahmin et
for ticker in tickers:
    data = yf.download(
        f'{ticker}.IS',
        start=start_date,
        end=end_date,
        # verileri aylık olarak çek
        interval="1mo",
        progress=False
    )

    data = data.reset_index()
    data = data[['Date', 'Adj Close']]

    data = data.dropna()

    first_data_date = data.iloc[0]['Date']
    # ilk tahmin verisetindeki ilk veri tarihinden 24 ay sonra başlar
    for i in range(24, len(data)):
        train_data = data[:i]
        test_data = data[i:i + 1]

        model = auto_arima(train_data['Adj Close'], max_order=None, stepwise=True)
        prediction = float(model.predict(n_periods=1, return_conf_int=False))

        temp_df = pd.DataFrame({
            'Date': [test_data['Date'].values[0]],
            'Ticker': [ticker],
            'Adj Close': [test_data['Adj Close'].values[0]],
            'Prediction': [prediction]
        })
        pred_df = pd.concat([pred_df, temp_df], ignore_index=True)

        print(f'{ticker}: {i}')
    pred_df.loc[pred_df['Ticker'] == ticker, 'First Data Date'] = first_data_date
    time.sleep(1)

# tahminlerin doğruluğunun değerlendirilmesi için root-mean-square error yani ortalama kare hatası
rmse_dict = {}

# hisse senetlerinin tahminlerini RMSE değerlerine göre sırala
for ticker in tickers:
    ticker_data = pred_df[pred_df['Ticker'] == ticker]

    actual_values = ticker_data['Adj Close'].values
    predicted_values = ticker_data['Prediction'].values

    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    rmse_dict[ticker] = rmse

# RMSE değerlerine göre hisse senetlerini sırala
sorted_rmse_dict = dict(sorted(rmse_dict.items(), key=lambda item: item[1]))

# RMSE değerlerini görselleştir
plt.figure(figsize=(12, 16))
# yatay çubuk grafiği
plt.barh(list(sorted_rmse_dict.keys()), list(sorted_rmse_dict.values()), color='red')
plt.xlabel('RMSE Value')
plt.ylabel('Stock Ticker')
plt.title('RMSE Values for Each Stock Ticker')
plt.tight_layout()
plt.show()
