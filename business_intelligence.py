# stock_dfs isimli klasörden fonkisyon girdisi olarak istediğim verinin kapanış değerlerini görselleştirme işlemi

import os
import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(stock):
    stock_dfs = os.listdir('stock_dfs')
    df = pd.read_csv("stock_dfs/" + stock + ".csv")
    df.set_index('Date', inplace=True)
    df['Close'].plot()
    plt.title(stock)
    plt.show()


visualize_data("AKBNK")
