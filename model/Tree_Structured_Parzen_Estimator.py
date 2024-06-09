# warningleri kaldırma kütüphanesi
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# warningleri kaldırma
warnings.filterwarnings('ignore')

# Dataframe'in daha anlaşılır görünebilmesi için
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Veri setini okuma
stock_path = "stock_dfs/AYDEM.csv"
stock = pd.read_csv(stock_path)

# artık date sütunu index olacak
stock.index = stock['Date']

# axis = 1 ile kolonları siler
stock = stock.drop(['Date'], axis=1)

# Close sütunundaki tüm verileri 1 ve 0 arasına normalize etme
# stock['Close'] = (stock['Close'] - stock['Close'].min()) / (stock['Close'].max() - stock['Close'].min())

# 70% training, 15% validation, 15% test
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(stock, target='Close',
                                                                            train_size=0.70,
                                                                            valid_size=0.15,
                                                                            test_size=0.15,
                                                                            method="sorted",
                                                                            sort_by_col="Date")

parameters = \
    {
        # n_estimators -> tahminci sayısı
        'n_estimators': [366],

        # learning_rate -> öğrenme oranı
        'learning_rate': [0.2733869934133383],

        # max_depth -> maksimum ağaç derinliği
        # Bir ağacın maksimum derinliğini artırmak,
        # modeli daha karmaşık hale getirecek ve overfitting olasılığını artıracaktır.
        'max_depth': [3],

        # gamma -> ağaçların ayrılması için gereken minimum kayıp
        'gamma': [0.009444434148380068],

        # min_child_weight -> ağaçların ayrılması için gereken minimum ağırlık
        'min_child_weight': [3],

        # subsample -> örneklem oranı
        'subsample': [0.3033676775175524],

        # colsample_bytree -> öznitelik örneklem oranı
        'colsample_bytree': [0.027011934674362817],

        # colsample_bylevel -> düzey örneklem oranı
        'colsample_bylevel': [0.04886581693206729]
    }

model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)
model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)

# Sonuçları Excel'e yazma
#pd.DataFrame(y_pred).to_excel('y_pred.xlsx', index=True)

# hata
mean_absolute_error(y_test, y_pred)

sns.lineplot(y=y_pred, x=np.arange(len(y_pred)), color='#243763', legend='full', label='Predicted')
sns.lineplot(y=y_test, x=np.arange(len(y_pred)), color='#FF6E31', legend='full', label='True')
plt.title('Y-True vs Y-Predicted')
plt.show()
