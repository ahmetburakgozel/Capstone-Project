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
stock_path = "yahoo_manual/AYDEM.IS.csv"
stock = pd.read_csv(stock_path)

# artık date sütunu index olacak
stock.index = stock['Date']

# axis = 1 ile kolonları siler
stock = stock.drop(['Date'], axis=1)

# head
stock.head()

# excele yazma
stock.to_excel("aydem.xlsx")

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
        'n_estimators': [500],

        # learning_rate -> öğrenme oranı
        'learning_rate': [0.1],

        # max_depth -> maksimum ağaç derinliği
        # Bir ağacın maksimum derinliğini artırmak,
        # modeli daha karmaşık hale getirecek ve overfitting olasılığını artıracaktır.
        'max_depth': [15],

        # gamma -> ağaçların ayrılması için gereken minimum kayıp
        'gamma': [0.01],

        # min_child_weight -> ağaçların ayrılması için gereken minimum ağırlık
        'min_child_weight': [4],

        # subsample -> örneklem oranı
        'subsample': [0.8],

        # colsample_bytree -> öznitelik örneklem oranı
        'colsample_bytree': [1],

        # colsample_bylevel -> düzey örneklem oranı
        'colsample_bylevel': [1]
    }

model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)
pd.DataFrame(y_pred).to_excel('y_pred.xlsx', index=True)

# hata
mean_absolute_error(y_test, y_pred)

sns.lineplot(y=y_pred, x=np.arange(len(y_pred)), color='#243763', legend='full', label='Predicted')
sns.lineplot(y=y_test, x=np.arange(len(y_pred)), color='#FF6E31', legend='full', label='True')
plt.title('Y-True vs Y-Predicted')
plt.show()
