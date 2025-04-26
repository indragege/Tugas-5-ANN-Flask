import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import pickle
import json

# Buat direktori yang diperlukan
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('static'):
    os.makedirs('static')

# Baca dataset dari file CSV
print("Membaca dataset kasus pneumonia Jawa Barat...")
df = pd.read_csv("dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv")

# Pilih data untuk Kota Bandung sebagai contoh
kota_bandung = df[df['nama_kabupaten_kota'] == 'KOTA BANDUNG'].copy()
data = {
    "Tahun": kota_bandung['tahun'].values,
    "Kasus": kota_bandung['jumlah_kasus'].values
}

# Konversi ke DataFrame
df_kasus = pd.DataFrame(data)

# Tampilkan 5 data pertama
print("\nData 5 tahun pertama untuk Kota Bandung:")
print(df_kasus.head())

# Visualisasi Data
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_kasus["Tahun"], y=df_kasus["Kasus"], color="blue", label="Data Aktual")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Kasus")
plt.title("Kasus Pneumonia di Kota Bandung")
plt.legend()
plt.savefig('static/data_aktual.png')
plt.close()

# Normalisasi data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_kasus[["Tahun", "Kasus"]])  # Normalisasi kedua kolom

# Pisahkan data menjadi input (X) dan output (Y)
X = df_scaled[:, 0].reshape(-1, 1)  # Tahun sebagai input
Y = df_scaled[:, 1]  # Kasus sebagai output

# Split data menjadi training dan testing set (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# Membangun model ANN
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),  # Hidden layer pertama
    Dense(10, activation='relu'),  # Hidden layer kedua
    Dense(1, activation='linear')  # Output layer
])

# Kompilasi model
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

# Melatih model
print("\nMelatih model...")
history = model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), verbose=1)

# Evaluasi model dengan data uji
loss, mae = model.evaluate(X_test, Y_test)
print(f"\nMean Absolute Error (MAE): {mae:.4f}")

# Prediksi jumlah kasus tahun 2024 dan 2025
tahun_prediksi = np.array([[2024], [2025]])
tahun_prediksi_scaled = scaler.transform(np.column_stack((tahun_prediksi, np.zeros(len(tahun_prediksi)))))[:, 0].reshape(-1, 1)

# Prediksi dengan model
prediksi_scaled = model.predict(tahun_prediksi_scaled)

# Balikkan skala hasil prediksi
prediksi = scaler.inverse_transform(np.column_stack((tahun_prediksi_scaled[:, 0], prediksi_scaled)))[:, 1]

print("\nHasil Prediksi:")
for tahun, kasus in zip([2024, 2025], prediksi):
    print(f"Prediksi jumlah kasus pneumonia di Kota Bandung tahun {tahun}: {int(kasus)} kasus")

# Prediksi untuk seluruh data uji
Y_pred = model.predict(X_test)

# Plot hasil prediksi vs data aktual
plt.figure(figsize=(8,5))
plt.scatter(scaler.inverse_transform(np.column_stack((X_test, np.zeros_like(X_test))))[:, 0], 
           scaler.inverse_transform(np.column_stack((np.zeros_like(Y_test), Y_test)))[:, 1], 
           color='blue', label="Data Aktual")
plt.scatter(scaler.inverse_transform(np.column_stack((X_test, np.zeros_like(X_test))))[:, 0], 
           scaler.inverse_transform(np.column_stack((np.zeros_like(Y_pred), Y_pred)))[:, 1], 
           color='red', label="Prediksi ANN")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Kasus")
plt.title("Hasil Prediksi vs Data Aktual Kasus Pneumonia di Kota Bandung")
plt.legend()
plt.savefig('static/hasil_prediksi.png')
plt.close()

# Simpan model untuk penggunaan di web
model.save('model/model_prediksi_pneumonia.h5')

# Simpan scaler untuk preprocessing data baru
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel dan scaler telah disimpan di folder 'model'")
print("Visualisasi telah disimpan di folder 'static'") 