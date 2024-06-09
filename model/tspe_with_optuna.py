import warnings

import numpy as np
import pandas as pd
from fast_ml.model_development import train_valid_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna

warnings.filterwarnings('ignore')

# Veri setini okuma
stock_path = "stock_dfs/HEKTS.csv"
stock = pd.read_csv(stock_path)

# artık date sütunu index olacak
stock.index = stock['Date']

# axis = 1 ile kolonları siler
stock = stock.drop(['Date'], axis=1)

# veri manipülasyonu volume sütununda 0 olan satırları silme
# borsanın kapalı/tatil olduğu günlerde volume 0 olabilir
stock = stock[stock['Volume'] != 0]

# aykırı değerleri bulma

# q1 -> 1. çeyrek
q1 = stock["Close"].quantile(0.25)
# q3 -> 3. çeyrek
q3 = stock["Close"].quantile(0.75)
# iqr -> interquartile range
iqr = q3 - q1
# up -> upper bound
up = q3 + 1.5 * iqr
# low -> lower bound
low = q1 - 1.5 * iqr

# aykırı değerleri bulma
outliers = stock[(stock["Close"] < low) | (stock["Close"] > up)].index
stock = stock.drop(outliers)

# Close sütunundaki tüm verileri 1 ve 0 arasına normalize etme
# stock['Close'] = (stock['Close'] - stock['Close'].min()) / (stock['Close'].max() - stock['Close'].min())

# 70% training, 15% validation, 15% test
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(stock, target='Close',
                                                                            train_size=0.70,
                                                                            valid_size=0.15,
                                                                            test_size=0.15,
                                                                            method="sorted",
                                                                            sort_by_col="Date")


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'colsample_bylevel': trial.suggest_loguniform('colsample_bylevel', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'rmse',
        'use_label_encoder': False
    }

    # modeli eğit
    optuna_model = XGBRegressor(**params)
    optuna_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=5, verbose=False)

    # Validation set üzerinde tahmin yap
    # çünkü modelin başarısını ölçmek için validation set kullanılır
    y_pred = optuna_model.predict(X_test)

    # tahminlerin doğruluğunu değerlendir
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Hataların Karesinin Ortalamasının Karekökü: %.2f" % rmse)

    return rmse


# Optuna örnekleyici oluştur, varsayılan TPESampler
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))
print('  Params: ')

# En iyi parametreleri yazdır
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# En iyi parametreler ile model oluşturma
params = trial.params
model = XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=True)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Hata hesaplama RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Hataların Karesinin Ortalamasının Karekökü: %.2f" % rmse)

# plot
sns.lineplot(y=y_pred, x=np.arange(len(y_pred)), color='#243763', legend='full', label='Tahmin Edilen Deger')
sns.lineplot(y=y_test, x=np.arange(len(y_pred)), color='#FF6E31', legend='full', label='Gercek Deger')
plt.title('Gercek vs Tahmin Edilen')
plt.show()

# Son veri noktasını al
last_data_point = X_test.iloc[-1].values.reshape(1, -1)


def predict_future(model, last_data_point, period):
    # Tahminlerin saklanacağı boş bir liste oluştur
    predictions = []

    # Verilen periyod boyunca tahmin yap
    for _ in range(period):
        # Bir sonraki zaman adımını tahmin et
        next_step_prediction = model.predict(last_data_point)
        # Tahmini listeye ekle
        predictions.append(next_step_prediction[0])

        # Son tahmini veri noktasına ekle ve bir sonraki tahmin için hazırla
        # Örneğin, son tahminin bir önceki günün kapanış fiyatı olduğunu varsayalım
        last_data_point = np.roll(last_data_point, -1)
        last_data_point[-1] = next_step_prediction

    return predictions


future_predictions = predict_future(model, last_data_point, 2)
future_predictions_df = pd.DataFrame(future_predictions, columns=['Close'])
print(future_predictions_df)

# # plot
# sns.lineplot(y=future_predictions, x=np.arange(len(future_predictions)), color='#243763', legend='full',
#              label='Tahmin Edilen Deger')
# plt.title('Tahmin Edilen')
# plt.show()
