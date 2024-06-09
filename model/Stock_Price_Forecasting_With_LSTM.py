import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# GPU'ların kullanılabilirliğini kontrol etme
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Veri setini yükle
df = pd.read_csv('stock_dfs/QUAGR.csv')

# Tarih sütununu datetime nesnesine dönüştür ve indeks yap
df['Date'] = pd.to_datetime(df['Date'])

# Tarih sütununu indeks yap
df.set_index('Date', inplace=True)

# volume sütununda 0 olan satırları silme
# borsanın kapalı/tatil olduğu günlerde volume 0 olabilir
df = df[df['Volume'] != 0]

# Verileri ön işleme tabi tutmak için Close sütununu al
data = df['Close'].values

# verileri ölçeklendir. tek satır çok sütunlu bir diziye dönüştür.
data = data.reshape(-1, 1)

# MinMaxScaler kullanarak verileri ölçeklendir (0-1 aralığına getir)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# create_dataset fonksiyonu veri setini zaman serisi veri setine dönüştürmek için kullanılır.
# Bu fonksiyon, veri setini X ve y olarak ikiye böler. time_step parametresi pencere boyutunu belirler.
# Örneğin, time_step=60 ise, her bir X verisi 60 veri içerir ve y verisi, X verisinin bir sonraki verisidir.
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


# Veri setini oluştur ve X, y olarak ayır
time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Verileri eğitim ve test setlerine ayır
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM modelini oluşturma

# Sıralı model oluştur
model = Sequential()

# İlk katman olarak Input nesnesini ekleyin ve pencere boyutunu belirtin
model.add(Input(shape=(time_step, 1)))

# LSTM katmanı ekle ve 50 nöron kullan  ve pencereyi döndür
model.add(LSTM(units=50, return_sequences=True))

# Dropout katmanı ekle ve %20 oranında nöronları devre dışı bırak (overfitting önlemek için)
model.add(Dropout(0.2))

# LSTM katmanı ekle ve 50 nöron kullan ve pencereyi döndürme
# burada return_sequences=False olacak çünkü bir sonraki LSTM katmanı yok
model.add(LSTM(units=50, return_sequences=False))

# Dropout katmanı ekle ve %20 oranında nöronları devre dışı bırak (overfitting önlemek için)
model.add(Dropout(0.2))

# Tam bağlı katman(fully connected) ekle ve 1 nöron kullan
model.add(Dense(1))

# Modeli derle ve kayıp fonksiyonu olarak ortalama kare hatayı kullan
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli özetle
model.summary()

"""
# ModelCheckpoint callback'ini oluştur
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
"""

# Modeli eğit
# epochs: Modelin kaç kez eğitileceğini belirler
# batch_size: Verinin kaç parçaya bölüneceğini belirler
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

"""
# En iyi modeli yükle
model.load_weights('best_model.keras')
"""

# Tahmin et ve değerleri ters dönüştür (normalizasyonu tersine çevir)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Ölçeklendirmeyi tersine çevir

# Sonuçları görselleştir
plt.figure(figsize=(14, 5))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue',
         label='Gerçek Hisse Fiyatı')
plt.plot(df.index[-len(y_test):], predictions, color='red',
         label='Tahmin Edilen Hisse Fiyatı')
plt.title('Hisse Fiyat Tahmini')  # Başlık
plt.xlabel('Tarih')  # X ekseninin başlığı
plt.ylabel('Hisse Fiyatı')  # Y ekseninin başlığı
plt.legend()
plt.show()

# Modeli değerlendir MSE - MAE - RMSE
mse = np.mean(np.square(predictions - scaler.inverse_transform(y_test.reshape(-1, 1))))
mae = np.mean(np.abs(predictions - scaler.inverse_transform(y_test.reshape(-1, 1))))
rmse = np.sqrt(mse)
print(f'Ortalama Kare Hata (MSE): {mse}')
print(f'Ortalama Mutlak Hata (MAE): {mae}')
print(f'Kök Ortalama Kare Hata (RMSE): {rmse}')

# loss ve val_loss değerlerini görselleştir
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()
