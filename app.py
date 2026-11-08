import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Membuka izin akses CORS agar file HTML bisa memanggil API

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Memuat file model machine learning...")
model_path = os.path.join(BASE_DIR, 'model_classifier_chains.joblib')
prep_path = os.path.join(BASE_DIR, 'preprocessor.joblib')
tau_path = os.path.join(BASE_DIR, 'crc_threshold.joblib')

model_cc = joblib.load(model_path)
preprocessor = joblib.load(prep_path)
threshold_tau = joblib.load(tau_path)

daftar_mapel = [
    'Bahasa Inggris Lanjut', 'Biologi', 'Ekonomi', 
    'Fisika', 'Geografi', 'Informatika', 
    'Kimia', 'Matematika Lanjut', 'Sosiologi'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_df = pd.DataFrame([{
            'MAT': float(data.get('MAT', 0)),
            'B.ING': float(data.get('B_ING', 0)),
            'INFOR': float(data.get('INFOR', 0)),
            'FIS': float(data.get('FIS', 0)),
            'KIM': float(data.get('KIM', 0)),
            'BIO': float(data.get('BIO', 0)),
            'EKO': float(data.get('EKO', 0)),
            'SOS': float(data.get('SOS', 0)),
            'GEO': float(data.get('GEO', 0)),
            'BAKAT': str(data.get('BAKAT', 'Tidak Ada')),
            'MINAT': str(data.get('MINAT', 'Tidak Ada')).capitalize()
        }])
        
        X_transformed = preprocessor.transform(input_df)
        probs = model_cc.predict_proba(X_transformed)[0]
        
        idx_rekomendasi = np.where(probs >= threshold_tau)[0]
        mapel_rekomendasi = [daftar_mapel[i] for i in idx_rekomendasi]
        
        mapel_detail = []
        for idx in range(len(daftar_mapel)):
            mapel_detail.append({
                'mapel': daftar_mapel[idx],
                'probabilitas': round(float(probs[idx]) * 100, 2),
                'is_recommended': bool(probs[idx] >= threshold_tau)
            })
            
        mapel_detail = sorted(mapel_detail, key=lambda x: x['probabilitas'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'threshold_tau': round(float(threshold_tau), 4),
            'rekomendasi_utama': mapel_rekomendasi,
            'detail_mapel': mapel_detail
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    print("--- API Server Siap Berjalan ---")
    app.run(debug=True, host='127.0.0.1', port=5000)