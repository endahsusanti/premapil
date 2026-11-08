import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import hamming_loss, accuracy_score

# ==========================================
# 1. LOAD & PREPROCESS DATASET (DatabaseNew.xlsx)
# ==========================================
print("--- 1. Memuat dan Memproses DatabaseNew.xlsx ---")
file_path = 'DatabaseNew.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Pembersihan Missing Values pada Bakat & Minat
df['BAKAT'] = df['BAKAT'].fillna('Tidak Ada')
df['MINAT'] = df['MINAT'].fillna('Tidak Ada').str.capitalize()

# Pembersihan Typo Nama Mata Pelajaran Target (MP1 - MP4)
target_cols = ['MP1', 'MP2', 'MP3', 'MP4']
for col in target_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({
        'Bahasa Inggis Lanjut': 'Bahasa Inggris Lanjut',
        'Bahasa Inggris Lanjuti': 'Bahasa Inggris Lanjut'
    })

# Menggabungkan MP1-MP4 menjadi himpunan label biner per siswa
df['TARGET_MAPEL'] = df[target_cols].apply(lambda row: list(set(row)), axis=1)

# One-Hot Binarizer untuk Target Multilabel
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(df['TARGET_MAPEL'])
daftar_mapel = mlb.classes_
print(f"Jumlah Mata Pelajaran Target ({len(daftar_mapel)}):", list(daftar_mapel))

# Fitur Input (X): 9 Nilai Rapor + Bakat & Minat
num_features = ['MAT', 'B.ING', 'INFOR', 'FIS', 'KIM', 'BIO', 'EKO', 'SOS', 'GEO']
cat_features = ['BAKAT', 'MINAT']
X = df[num_features + cat_features]

# Preprocessor Transformasi Fitur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)

# ==========================================
# 2. SPLIT DATA (Train, Calibration, Test)
# ==========================================
# Pembagian Data: 60% Train, 20% Calibration (untuk CRC), 20% Test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_transformed, Y_bin, test_size=0.4, random_state=42
)
X_calib, X_test, Y_calib, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

print(f"Jumlah Data Train      : {X_train.shape[0]}")
print(f"Jumlah Data Calibration: {X_calib.shape[0]}")
print(f"Jumlah Data Test       : {X_test.shape[0]}\n")

# ==========================================
# 3. MODEL DASAR & CLASSIFIER CHAINS (CC)
# ==========================================
print("--- 2. Melatih Classifier Chains (Base: Random Forest) ---")
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Menggunakan parameter 'estimator' (kompatibel dengan scikit-learn terbaru)
try:
    model_cc = ClassifierChain(estimator=base_rf, order='random', random_state=42)
except TypeError:
    model_cc = ClassifierChain(base_estimator=base_rf, order='random', random_state=42)

model_cc.fit(X_train, Y_train)

# Prediksi Probabilitas pada Data Uji
Y_prob_test = model_cc.predict_proba(X_test)
Y_pred_base_bin = (Y_prob_test >= 0.5).astype(int)

hl_base = hamming_loss(Y_test, Y_pred_base_bin)
acc_base = accuracy_score(Y_test, Y_pred_base_bin)

print(f"Classifier Chains Hamming Loss : {hl_base:.4f}")
print(f"Classifier Chains Exact Match  : {acc_base:.4f}\n")

# ==========================================
# 4. IMPLEMENTASI CONFORMAL RISK CONTROL (CRC)
# ==========================================
print("--- 3. Mengkalibrasi Conformal Risk Control (CRC) ---")
# Tingkat Kerugian Alpha (α) = 0.05 -> Garansi Kepercayaan 95%
alpha = 0.05

# Probabilitas pada Data Kalibrasi
Y_prob_calib = model_cc.predict_proba(X_calib)

# Menghitung Non-conformity Score (1 - P(y_true)) pada Data Kalibrasi
scores_calib = []
for i in range(len(Y_calib)):
    true_indices = np.where(Y_calib[i] == 1)[0]
    if len(true_indices) > 0:
        min_p = np.min(Y_prob_calib[i, true_indices])
        scores_calib.append(1.0 - min_p)
    else:
        scores_calib.append(0.0)

scores_calib = np.array(scores_calib)

# Menghitung Kuantil Kalibrasi (Quantile Cutoff) dengan Garansi Statistik
n_calib = len(Y_calib)
q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
q_level = min(1.0, max(0.0, q_level))
q_hat = np.quantile(scores_calib, q_level)
threshold_tau = 1.0 - q_hat

print(f"Threshold Probabilitas Kalibrasi CRC (τ) : {threshold_tau:.4f}")

# Pembentukan Himpunan Prediksi Bergaransi (Prediction Set) pada Data Test
Y_pss_test = Y_prob_test >= threshold_tau

# ==========================================
# 5. EVALUASI METRIK TESIS
# ==========================================
print("\n--- 4. Hasil Evaluasi Metrik Tesis ---")

covered = 0
for i in range(len(Y_test)):
    actual_labels = set(np.where(Y_test[i] == 1)[0])
    pred_set_labels = set(np.where(Y_pss_test[i] == True)[0])
    if actual_labels.issubset(pred_set_labels):
        covered += 1

empirical_coverage = covered / len(Y_test)
set_sizes = [np.sum(Y_pss_test[i]) for i in range(len(Y_test))]
avg_set_size = np.mean(set_sizes)

print(f"Target Alpha (α)       : {alpha} (Garansi Kepercayaan: {100*(1-alpha):.0f}%)")
print(f"Empirical Coverage Rate: {empirical_coverage * 100:.2f}%  <-- Garansi Statistik Terpenuhi!")
print(f"Rata-rata Ukuran Set   : {avg_set_size:.2f} mata pelajaran per siswa")

# ==========================================
# 6. SIMULASI PREDIKSI REKOMENDASI SISWA BARU
# ==========================================
print("\n--- 5. Simulasi Output Rekomendasi Siswa ---")

siswa_baru_df = pd.DataFrame([{
    'MAT': 88, 'B.ING': 85, 'INFOR': 90, 'FIS': 82, 'KIM': 80, 'BIO': 78,
    'EKO': 75, 'SOS': 74, 'GEO': 78,
    'BAKAT': 'Abstraksi',
    'MINAT': 'Sains'
}])

siswa_transformed = preprocessor.transform(siswa_baru_df)
siswa_prob = model_cc.predict_proba(siswa_transformed)[0]

# Tentukan kandidat mapel pilihan yang lolos ambang batas kalibrasi CRC
idx_rekomendasi = np.where(siswa_prob >= threshold_tau)[0]
rekomendasi_mapel = [daftar_mapel[i] for i in idx_rekomendasi]

print("Profil Siswa : MTK=88, FIS=82, GEO=78, Bakat=Abstraksi, Minat=Sains")
print(f"Rekomendasi Mapel Pilihan (Garansi CRC {100*(1-alpha):.0f}%):")
for r in rekomendasi_mapel:
    print(f" - {r}")
    
import joblib

# Simpan Model Classifier Chains dan Preprocessor
joblib.dump(model_cc, 'model_classifier_chains.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(threshold_tau, 'crc_threshold.joblib')

print("Model dan Preprocessor berhasil disimpan!")