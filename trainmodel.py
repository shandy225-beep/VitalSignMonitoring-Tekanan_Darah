import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Pastikan file pan_tompkins_plus_plus_v2.py ada di dalam folder 'algo'
try:
    from algo.pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus
except ImportError:
    print("Error: Pan_Tompkins_Plus_Plus tidak ditemukan!")

# =====================================================================
# 1. KELAS EKSTRAKSI FITUR (10 Fitur Skripsi)
# =====================================================================
class FeatureExtractorSkripsi:
    def __init__(self, fs):
        self.fs = fs

    def compute(self, ecg, r_peaks):
        fs = self.fs
        N = len(ecg)
        r_peaks = np.asarray([p for p in r_peaks if 0 < p < N - 1], int)
        if N < fs * 3 or len(r_peaks) < 3: return None

        rms = float(np.sqrt(np.mean(ecg ** 2)))
        
        rr = np.diff(r_peaks) / fs
        if rr.size == 0: return None
        mean_rr = float(np.mean(rr))
        hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
        hrv = float(np.std(rr) * 1000.0)

        qt_list, tq_list = [], []
        for i in range(len(r_peaks) - 1):
            r1, r2 = r_peaks[i], r_peaks[i + 1]
            s0, s1 = r1 + int(0.15 * fs), r1 + int(0.40 * fs)
            s0, s1 = max(s0, 0), min(s1, N)
            if s1 <= s0: continue
            seg = ecg[s0:s1]
            t_rel = np.argmax(seg)
            t_idx = s0 + t_rel
            qt, tq = (t_idx - r1) / fs, (r2 - t_idx) / fs
            if qt > 0 and tq > 0:
                qt_list.append(qt); tq_list.append(tq)

        qt_arr, tq_arr = np.array(qt_list), np.array(tq_list)
        mean_qt = float(np.mean(qt_arr)) if qt_arr.size else np.nan
        mean_tq = float(np.mean(tq_arr)) if tq_arr.size else np.nan
        qtc_bazett = float(mean_qt / np.sqrt(mean_rr)) if mean_qt > 0 and mean_rr > 0 else np.nan

        diff1, diff2 = np.diff(ecg), np.diff(np.diff(ecg))
        var0, var1, var2 = np.var(ecg), np.var(diff1), np.var(diff2)
        hj_mob = float(np.sqrt(var1 / var0)) if var0 > 0 else np.nan
        hj_comp = float(np.sqrt(var2 / var1)) if var1 > 0 else np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_qt_tq = qt_arr / tq_arr if qt_arr.size and tq_arr.size else []
            ratio_qt_rr = qt_arr / rr[:len(qt_arr)] if rr.size else []

        sdi_qt_tq = float(np.nanstd(ratio_qt_tq)) if len(ratio_qt_tq) else np.nan
        sdin_qt_rr = float(np.nanstd(ratio_qt_rr)) if len(ratio_qt_rr) else np.nan

        return {
            "Mean_QT": mean_qt, "TQ": mean_tq, "QTc_Bazett": qtc_bazett,
            "Hjorth_Mobility": hj_mob, "Hjorth_Complexity": hj_comp,
            "RMS": rms, "SDI": sdi_qt_tq, "SDIn": sdin_qt_rr,
            "HR": hr, "HRV": hrv
        }

# =====================================================================
# 2. PROSES PEMBUATAN DATASET
# =====================================================================
print("Memulai Preprocessing...")
df = pd.read_csv('df.csv')
fs = 125 

def apply_filters(ecg_raw, fs):
    ecg = ecg_raw - np.mean(ecg_raw)
    b, a = butter(2, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    ecg = filtfilt(b, a, ecg)
    bn, an = iirnotch(50.0 / (fs / 2), Q=30)
    ecg = filtfilt(bn, an, ecg)
    return ecg

extractor = FeatureExtractorSkripsi(fs)
pt_detector = Pan_Tompkins_Plus_Plus()

epoch_sec = 20
samples_per_epoch = fs * epoch_sec
kernel = int(0.2 * fs)
if kernel % 2 == 0: kernel += 1

num_epochs = len(df) // samples_per_epoch
features_list = []

for i in range(num_epochs):
    start, end = i * samples_per_epoch, (i + 1) * samples_per_epoch
    ecg_seg = df['ECG'].iloc[start:end].values
    abp_seg = df['ABP'].iloc[start:end].values
    
    ecg_clean = apply_filters(ecg_seg, fs)
    try:
        r_peaks = pt_detector.rpeak_detection(ecg_clean, fs)
    except: continue
        
    if len(r_peaks) < 3: continue
    feat = extractor.compute(ecg_clean, r_peaks)
    if feat is None: continue
        
    abp_smooth = medfilt(abp_seg, kernel_size=kernel)
    locs_s, _ = find_peaks(abp_smooth, distance=int(0.25*fs))
    locs_d = []
    for j in range(len(locs_s) - 1):
        idx_min = np.argmin(abp_smooth[locs_s[j]:locs_s[j+1]])
        locs_d.append(locs_s[j] + idx_min)
        
    if len(locs_s) < 2 or len(locs_d) < 1: continue
    sbp_val, dbp_val = np.mean(abp_smooth[locs_s]), np.mean(abp_smooth[locs_d])
    if sbp_val < 50 or dbp_val < 30: continue 
        
    feat["systolic"] = sbp_val
    feat["diastolic"] = dbp_val
    features_list.append(feat)

epoch_df = pd.DataFrame(features_list).dropna()

# =====================================================================
# 3. PELATIHAN MODEL RANDOM FOREST
# =====================================================================
print("Melatih Model Random Forest...")
feature_columns = [
    "Mean_QT", "TQ", "QTc_Bazett", "Hjorth_Mobility", "Hjorth_Complexity", 
    "RMS", "SDI", "SDIn", "HR", "HRV"
]

X = epoch_df[feature_columns].values
y_sbp, y_dbp = epoch_df["systolic"].values, epoch_df["diastolic"].values

X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(X, y_sbp, test_size=0.2, random_state=42)
_, _, y_dbp_train, y_dbp_test = train_test_split(X, y_dbp, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

if not os.path.exists('model'): os.makedirs('model')
joblib.dump(scaler, 'model/feat_scaler.pkl')

rf_sbp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_sbp.fit(X_train_scaled, y_sbp_train)
joblib.dump(rf_sbp, 'model/rf_sbp_N.pkl')

rf_dbp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_dbp.fit(X_train_scaled, y_dbp_train)
joblib.dump(rf_dbp, 'model/rf_dbp_N.pkl')

print("Pelatihan Selesai! Model berhasil disimpan.")