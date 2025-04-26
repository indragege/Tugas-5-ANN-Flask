from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load dataset
print("Loading dataset pneumonia Jawa Barat...")
df = pd.read_csv('dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv')

# Load scalers
with open('model/scaler.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Load model info
with open('static/model_info.json', 'r') as f:
    model_info = json.load(f)

# Siapkan data untuk dropdown
regions = []
for kode in df['kode_kabupaten_kota'].unique():
    kode_str = str(kode)
    if kode_str in model_info:
        info = model_info[kode_str]
        
        # Baca ringkasan model
        try:
            with open(f'static/model_summary_{kode}.txt', 'r') as f:
                model_summary = f.read()
        except:
            model_summary = "Model summary tidak tersedia"
        
        regions.append({
            'kode': kode_str,
            'nama': info['nama'],
            'stats': info['stats'],
            'metrics': info['metrics'],
            'model_summary': model_summary
        })

regions = sorted(regions, key=lambda x: x['nama'])

# Load model dan scaler
model = load_model('model/model_prediksi_pneumonia.h5')
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset untuk mendapatkan data historis
df = pd.read_csv("dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv")
kota_bandung = df[df['nama_kabupaten_kota'] == 'KOTA BANDUNG'].copy()

# Buat plot awal
plt.figure(figsize=(10, 6))
sns.scatterplot(data=kota_bandung, x='tahun', y='jumlah_kasus', color='blue', label='Data Aktual')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Kasus')
plt.title('Kasus Pneumonia di Kota Bandung')
plt.legend()
plt.savefig('static/data_aktual.png')
plt.close()

@app.route('/')
def home():
    return render_template('index.html', regions=regions)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input tahun dari form
        tahun = int(request.form['tahun'])
        
        # Validasi tahun
        if tahun < 2024:
            return jsonify({
                'error': 'Tahun prediksi harus 2024 atau lebih besar'
            })
        
        # Persiapkan data untuk prediksi
        tahun_scaled = scaler.transform([[tahun, 0]])[:, 0].reshape(-1, 1)
        
        # Lakukan prediksi
        prediksi_scaled = model.predict(tahun_scaled)
        
        # Balikkan skala hasil prediksi
        prediksi = scaler.inverse_transform(
            np.column_stack((tahun_scaled[:, 0], prediksi_scaled)))[:, 1]
        
        # Buat plot baru dengan hasil prediksi
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=kota_bandung, x='tahun', y='jumlah_kasus', color='blue', label='Data Aktual')
        plt.scatter(tahun, prediksi[0], color='red', label='Prediksi', s=100)
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Kasus')
        plt.title('Kasus Pneumonia di Kota Bandung')
        plt.legend()
        plt.savefig('static/hasil_prediksi.png')
        plt.close()
        
        return jsonify({
            'tahun': tahun,
            'prediksi': int(prediksi[0]),
            'plot_url': 'static/hasil_prediksi.png'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Terjadi kesalahan: {str(e)}'
        })

@app.route('/model_info/<kode>')
def get_model_info(kode):
    try:
        region = next((r for r in regions if r['kode'] == kode), None)
        
        if not region:
            return jsonify({
                'success': False,
                'message': 'Kabupaten/kota tidak ditemukan'
            })
            
        return jsonify({
            'success': True,
            'data': {
                'nama': region['nama'],
                'stats': region['stats'],
                'metrics': region['metrics'],
                'model_summary': region['model_summary']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Terjadi kesalahan: {str(e)}'
        })

if __name__ == '__main__':
    print("\nInformasi Dataset:")
    print("Dataset ini berisi jumlah kasus penyakit pneumonia berdasarkan kabupaten/kota")
    print("di Provinsi Jawa Barat dari tahun 2019 s.d. 2023")
    print("Dataset dihasilkan oleh Dinas Kesehatan (periode 1 tahun sekali)")
    print("\nServer berjalan di http://localhost:5000")
    app.run(debug=True) 