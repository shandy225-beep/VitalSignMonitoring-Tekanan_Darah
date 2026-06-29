import json, joblib
import sys
import serial
import serial.tools.list_ports
import numpy as np
import os
import math
import time
from collections import deque
from joblib import load as joblib_load
from scipy.signal import butter, filtfilt, iirnotch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QFrame, QPushButton, QComboBox, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg

from PyQt5.QtCore import QThread, pyqtSignal

from algo.pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus


from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
SHIMMER_AVAILABLE = True

# KONFIG DASAR
FS = 125
STREAM_DT = 0.1                   # update GUI tiap 100 ms
BUFFER_SECONDS = 30               # buffer 30 detik
FEATURE_WINDOW_SEC = 20           # window fitur 20 detik
PREDICT_INTERVAL = 10.0           # prediksi tiap 10 detik

RESULTS_DIR = "results"
MODELS_DIR = "model"
SBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_sbp_revisi2.pkl")
DBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_dbp_revisi2.pkl")
FEATURE_ORDER_JSON = os.path.join(MODELS_DIR, "features_order2.json")

PORT = "COM6"                  

# ------------------------------------------------------------
# STATISTIK FITUR TRAINING (MIT-BIH) - hasil dari ECG_features_summary.csv
# ------------------------------------------------------------
# mean/std tiap fitur di domain training (MIT-BIH/Kaggle-style)
MITBIH_STATS = {
    "Mean_QT(s)":      {"mean": 0.16677517628000713, "std": 0.06299517293060142},
    "TQ(s)":           {"mean": 0.5694139594114122,  "std": 0.15839566289940465},
    "QTc_Bazett(s)":   {"mean": 0.19947462373196448, "std": 0.08778862864971136},
    "Hjorth_Mobility": {"mean": 0.5540765718546059,  "std": 0.1957138682586237},
    "Hjorth_Complexity": {"mean": 1.894426795663087, "std": 0.43456218794481016},
    "RMS":             {"mean": 0.557967122217131,   "std": 0.2654334826537493},
    "SDI(QT/TQ)":      {"mean": 0.3814838919454695,  "std": 0.5107865234766669},
    "SDIn(QT/RR)":     {"mean": 0.24078524359236897, "std": 0.126874361432792},
    "HR(bpm)":         {"mean": 84.00825367532502,   "std": 15.599935239848898},
    # HRV(ms): aproksimasi fisiologis
    "HRV(ms)":         {"mean": 60.0,                "std": 25.0},
}

# jumlah minimal window prediksi untuk kalibrasi domain Shimmer
DA_MIN_SAMPLES = 15 

# R-PEAK REFINEMENT
def refine_rpeaks(sig_norm, r_peaks, fs, min_rr_ms=280, amp_thr_rel=0.30):
    r_peaks = np.asarray(r_peaks, dtype=int)
    if r_peaks.size == 0:
        return r_peaks

    amps = np.abs(sig_norm[r_peaks])
    max_amp = np.max(amps)
    if max_amp <= 0:
        return np.array([], dtype=int)

    amp_thr = amp_thr_rel * max_amp
    cand = r_peaks[amps >= amp_thr]
    if cand.size == 0:
        return np.array([r_peaks[np.argmax(amps)]], dtype=int)

    cand = np.sort(cand)
    min_dist = int((min_rr_ms / 1000.0) * fs)

    refined = [cand[0]]
    for p in cand[1:]:
        if p - refined[-1] < min_dist:
            if np.abs(sig_norm[p]) > np.abs(sig_norm[refined[-1]]):
                refined[-1] = p
        else:
            refined.append(p)

    return np.asarray(refined, dtype=int)

# EKSTRAKTOR FITUR (kompatibel dengan prediksi_revisi)
class FeatureExtractorRevisi:
    def __init__(self, fs):
        self.fs = fs

    def compute(self, ecg, r_peaks):
        fs = self.fs
        ecg = np.asarray(ecg, float)
        N = len(ecg)

        r_peaks = np.asarray([p for p in r_peaks if 0 < p < N - 1], int)
        if N < fs * 3 or len(r_peaks) < 3:
            return None

        # RMS
        rms = float(np.sqrt(np.mean(ecg ** 2)))

        # RR interval
        rr = np.diff(r_peaks) / fs
        if len(rr) == 0:
            return None
        mean_rr = float(np.nanmean(rr))
        hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
        hrv = float(np.std(rr) * 1000.0) if len(rr) > 1 else np.nan

        # QT & TQ (deteksi T sederhana)
        qt_list, tq_list = [], []
        for i in range(len(r_peaks) - 1):
            r1, r2 = r_peaks[i], r_peaks[i + 1]

            search_start = r1 + int(0.15 * fs)
            search_stop = r1 + int(0.4 * fs)
            search_start = max(search_start, 0)
            search_stop = min(search_stop, N)
            if search_stop <= search_start:
                continue

            seg = ecg[search_start:search_stop]
            t_rel = int(np.argmax(seg))
            t_idx = search_start + t_rel

            qt = (t_idx - r1) / fs
            tq = (r2 - t_idx) / fs

            if qt > 0 and tq > 0:
                qt_list.append(qt)
                tq_list.append(tq)

        qt_arr = np.array(qt_list)
        tq_arr = np.array(tq_list)

        mean_qt = float(np.nanmean(qt_arr)) if qt_arr.size else np.nan
        mean_tq = float(np.nanmean(tq_arr)) if tq_arr.size else np.nan

        # QTc Bazett
        if mean_qt > 0 and mean_rr > 0:
            qtc_bazett = float(mean_qt / np.sqrt(mean_rr))
        else:
            qtc_bazett = np.nan

        # Hjorth parameters
        diff1 = np.diff(ecg)
        diff2 = np.diff(diff1)
        var0 = float(np.var(ecg))
        var1 = float(np.var(diff1))
        var2 = float(np.var(diff2))

        if var0 > 0:
            hj_mob = float(np.sqrt(var1 / var0))
        else:
            hj_mob = np.nan

        if var1 > 0:
            hj_comp = float(np.sqrt(var2 / var1))
        else:
            hj_comp = np.nan

        # SDI rasio QT/TQ, QT/RR
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_qt_tq = qt_arr / tq_arr if (qt_arr.size and tq_arr.size) else np.array([])
            ratio_qt_rr = qt_arr / rr[:len(qt_arr)] if rr.size else np.array([])

        sdi_qt_tq = float(np.nanstd(ratio_qt_tq)) if ratio_qt_tq.size else np.nan
        sdin_qt_rr = float(np.nanstd(ratio_qt_rr)) if ratio_qt_rr.size else np.nan

        feat = {
            "Mean_QT(s)": mean_qt,
            "TQ(s)": mean_tq,
            "QTc_Bazett(s)": qtc_bazett,
            "Hjorth_Mobility": hj_mob,
            "Hjorth_Complexity": hj_comp,
            "RMS": rms,
            "SDI(QT/TQ)": sdi_qt_tq,
            "SDIn(QT/RR)": sdin_qt_rr,
            "HR(bpm)": float(hr),
            "HRV(ms)": hrv,
        }
        return feat


# PREDICTOR GPR (mean ± std) + KALIBRASI OUTPUT
class PredictorRevisi:
    def __init__(self):
        self.model_sbp = joblib_load(SBP_MODEL_PKL)
        self.model_dbp = joblib_load(DBP_MODEL_PKL)

        with open(FEATURE_ORDER_JSON, "r") as f:
            self.feature_order = json.load(f)

        print("[INFO] Fitur yang dipakai model:", self.feature_order)

        # --- load calibration.json kalau ada ---
        self.calib = None
        CALIB_PATH = os.path.join("models", "calibration.json")

        try:
            with open(CALIB_PATH, "r") as f:
                self.calib = json.load(f)
            print(f"[INFO] calibration.json ditemukan di: {CALIB_PATH}")
        except FileNotFoundError:
            print(f"[INFO] calibration.json TIDAK ditemukan di: {CALIB_PATH}")
        except Exception as e:
            print(f"[WARN] Gagal membaca calibration.json: {e}")
            self.calib = None

    def predict(self, feat_dict):
        try:
            X = np.array([[feat_dict[k] for k in self.feature_order]], dtype=float)
        except KeyError as e:
            return None, None, None, None, f"Fitur hilang: {e}"

        if np.any(~np.isfinite(X)):
            return None, None, None, None, "Ada fitur NaN/inf"

        try:
            sbp_mean, sbp_std = self.model_sbp.predict(X, return_std=True)
            dbp_mean, dbp_std = self.model_dbp.predict(X, return_std=True)

            sbp_mean = float(sbp_mean[0])
            sbp_std = float(sbp_std[0])
            dbp_mean = float(dbp_mean[0])
            dbp_std = float(dbp_std[0])

            # --- TERAPKAN KALIBRASI OUTPUT (jika tersedia) ---
            if self.calib is not None:
                try:
                    a = self.calib["SBP"]["a"]
                    b = self.calib["SBP"]["b"]
                    c = self.calib["DBP"]["c"]
                    d = self.calib["DBP"]["d"]

                    sbp_mean = a * sbp_mean + b
                    dbp_mean = c * dbp_mean + d
                except Exception as e:
                    print(f"[WARN] Gagal mengaplikasikan kalibrasi output: {e}")

            return sbp_mean, sbp_std, dbp_mean, dbp_std, None

        except Exception as e:
            return None, None, None, None, str(e)

class ShimmerReader(QThread):
    new_data = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()


    def __init__(self, port, baudrate=DEFAULT_BAUDRATE if SHIMMER_AVAILABLE else 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.shim_dev = None
    
    def adc_to_milivotls(self, adc_signal, gain=6, offset=0, vref=2.42,adc_bits=24):
        adc_sensitivity = (vref * 1000) / (2 ** (adc_bits - 1) - 1)
        return ((adc_signal - offset) * adc_sensitivity)/gain


    def handle_packet(self, pkt: DataPacket):
        try:
            # for channel in pkt._values:
            #     print(f"Channel: {channel}, Value: {pkt._values[channel]}")
            if EChannelType.EXG_ADS1292R_2_CH1_24BIT in pkt._values:
                ecg_raw = pkt[EChannelType.EXG_ADS1292R_2_CH1_24BIT]
                ecg_mv = ShimmerReader.adc_to_milivotls(self, ecg_raw, gain=6, offset=0)
                # emit latest value
                self.new_data.emit(ecg_mv)
        except Exception as e:
            print(f"Error handling packet: {e}")

    def run(self):  
        try:
            serial_conn = serial.Serial(self.port, self.baudrate, rtscts=False, dsrdtr=False)
            self.shim_dev = ShimmerBluetooth(serial_conn)
            self.shim_dev.initialize()
            dev_name = self.shim_dev.get_device_name()
            print(f"Connected to Shimmer: {dev_name}")

            self.shim_dev.add_stream_callback(self.handle_packet)
            self.shim_dev.start_streaming()
            try:
                self.connected_signal.emit()
            except Exception as e:
                self.disconnected_signal.emit()

            # main loop
            while self._running:
                time.sleep(0.05)

            # keluar
            print("Stopping shimmer...")

            try:
                self.shim_dev.stop_streaming()
            except Exception as e:
                print("stop_streaming error:", e)

            try:
                self.shim_dev.shutdown()
            except Exception as e:
                print("shutdown error:", e)

            try:
                serial_conn.close()
            except:
                pass

            print("Thread ended cleanly.")

        except Exception as e:
            error_msg = f"Shimmer connection error: {e}"
            print(error_msg)
            self.error_signal.emit(error_msg)


    def stop(self):
        self._running = False
        try:
            if self.shim_dev:
                self.shim_dev.stop_streaming()
                self.shim_dev.shutdown()
        except:
            pass


class SerialReader(QThread):
    new_data = pyqtSignal(float)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            while self._running:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    try:
                        val = float(line)
                        self.new_data.emit(val)
                    except:
                        pass
            ser.close()
        except Exception as e:
            print("Serial error:", e)

    def stop(self):
        self._running = False


class ECG_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Single-Channel Wearable ECG")
        self.resize(1200, 700)
        self.setStyleSheet("""
            QWidget { background-color: #fafafa; font-family: Segoe UI; color: #222; }
            QLabel { font-size: 14px; }
            QPushButton {
                background-color: #1976D2; color: white; border-radius: 6px;
                padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1565C0; }
            QComboBox { padding: 4px; background-color: white; border: 1px solid #ccc; border-radius: 4px; }
        """)

        self.detector = Pan_Tompkins_Plus_Plus()
        self.extractor = FeatureExtractorRevisi(FS)
        self.predictor = PredictorRevisi()

        self.sbp_hist = deque(maxlen=3)
        self.dbp_hist = deque(maxlen=3)
        self.hr_hist = deque(maxlen=3)
        self.hrv_hist = deque(maxlen=3)

        # ---------- DOMAIN ADAPTATION STATE ----------
        self.da_stats = {
            name: {"n": 0, "mean": 0.0, "M2": 0.0}
            for name in MITBIH_STATS.keys()
        }
        self.da_ready = False
        self.da_total_windows = 0

        central = QWidget()
        main_layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        header = QHBoxLayout()
        header.setSpacing(20)

        left_logos = QHBoxLayout()

        self.logo_univ = QLabel()
        if os.path.exists("./assets/logoUB.png"):
            self.logo_univ.setPixmap(QPixmap("./assets/logoUB.png").scaled(80, 80, Qt.KeepAspectRatio))
        else:
            self.logo_univ.setText("Logo UB\nNot Found")

        self.logo_rssa = QLabel()
        if os.path.exists("./assets/RSSA.png"):
            self.logo_rssa.setPixmap(QPixmap("./assets/RSSA.png").scaled(80, 80, Qt.KeepAspectRatio))
        else:
            self.logo_rssa.setText("Logo RSSA\nNot Found")

        left_logos.addWidget(self.logo_univ)
        left_logos.addWidget(self.logo_rssa)

        header.addLayout(left_logos)


        # ====== TITLE  ======
        title = QLabel("Single-Channel Wearable ECG sebagai Alternatif Patient Vital Sign Monitor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #1976D2;")
        header.addWidget(title, stretch=1)

        right_logos = QHBoxLayout()

        self.logo_filkom = QLabel()
        if os.path.exists("./assets/logoFILKOM.png"):
            self.logo_filkom.setPixmap(QPixmap("./assets/logoFILKOM.png").scaled(150, 70, Qt.KeepAspectRatio))

        self.logo_lab = QLabel()
        if os.path.exists("./assets/logoRES.png"):
            self.logo_lab.setPixmap(QPixmap("./assets/logoRES.png").scaled(70, 70, Qt.KeepAspectRatio))

        right_logos.addWidget(self.logo_filkom)
        right_logos.addWidget(self.logo_lab)

        header.addLayout(right_logos)

        main_layout.addLayout(header)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        left_panel = QVBoxLayout()
        self.pg_plot = pg.PlotWidget(title="Sinyal ECG Real-time")
        self.pg_plot.setBackground('w')
        self.pg_plot.showGrid(x=True, y=True, alpha=0.1)
        self.pg_plot.setLabel('left', 'Amplitudo (mV)')
        self.pg_plot.setLabel('bottom', 'Waktu (s)')
        self.pg_plot.setYRange(-1.5, 1.5)
        self.pg_plot.setXRange(-10, 0)
        self.ecg_curve = self.pg_plot.plot(pen=pg.mkPen(color='#1976D2', width=2))
        left_panel.addWidget(self.pg_plot, stretch=1)

        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.status_label = QLabel("Not Connected")
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_stream)
        self.stop_btn.setEnabled(False)

        self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#b71c1c;")
        port_layout.addWidget(self.status_label)
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.refresh_btn)
        port_layout.addWidget(self.start_btn)
        port_layout.addWidget(self.stop_btn)
        left_panel.addLayout(port_layout)

        content_layout.addLayout(left_panel, 2)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)

        def create_card(title_text, icon, unit_text=""):
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border-radius: 12px;
                    border: 1px solid #ddd;
                    padding: 12px;
                }
            """)
            layout = QVBoxLayout(frame)
            title_lbl = QLabel(f"{icon}  {title_text}")
            title_lbl.setStyleSheet("font-size: 12pt; font-weight: bold; color: #1976D2;")
            value_lbl = QLabel("--")
            value_lbl.setAlignment(Qt.AlignCenter)
            value_lbl.setStyleSheet("font-size: 32pt; font-weight: bold; color: #333;")
            unit_lbl = QLabel(unit_text)
            unit_lbl.setAlignment(Qt.AlignCenter)
            unit_lbl.setStyleSheet("font-size: 10pt; color: gray;")
            layout.addWidget(title_lbl)
            layout.addWidget(value_lbl)
            layout.addWidget(unit_lbl)
            return frame, value_lbl
        
        def create_bp_card(title_text, icon, unit_text=""):
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border-radius: 12px;
                    border: 1px solid #ddd;
                    padding: 10px;
                }
            """)
            layout = QVBoxLayout(frame)

            title_lbl = QLabel(f"{icon}  {title_text}")
            title_lbl.setStyleSheet("font-size: 12pt; font-weight: bold; color: #1976D2;")

            value_lbl = QLabel("--")
            value_lbl.setAlignment(Qt.AlignCenter)
            value_lbl.setStyleSheet("font-size: 32pt; font-weight: bold; color: #333;")

            std_layout = QHBoxLayout()

            std_min_lbl = QLabel("min: --/--")
            std_min_lbl.setAlignment(Qt.AlignLeft)
            std_min_lbl.setStyleSheet("font-size: 10pt; color: #666;")

            std_max_lbl = QLabel("max: --/--")
            std_max_lbl.setAlignment(Qt.AlignRight)
            std_max_lbl.setStyleSheet("font-size: 10pt; color: #666;")

            std_layout.addWidget(std_min_lbl)
            std_layout.addWidget(std_max_lbl)

            unit_lbl = QLabel(unit_text)
            unit_lbl.setAlignment(Qt.AlignCenter)
            unit_lbl.setStyleSheet("font-size: 10pt; color: gray;")

            layout.addWidget(title_lbl)
            layout.addWidget(value_lbl)
            layout.addLayout(std_layout)
            layout.addWidget(unit_lbl)

            return frame, value_lbl, std_min_lbl, std_max_lbl


        hr_frame, self.hr_label = create_card("Heart Rate", "❤️", "BPM")
        right_panel.addWidget(hr_frame)
        rr_frame, self.resp_label = create_card("Respiratory Rate", "🌬", "breaths/min")
        right_panel.addWidget(rr_frame)
        bp_frame, self.bp_label, self.bp_std_min_label, self.bp_std_max_label = create_bp_card("Blood Pressure", "🩸", "mmHg")
        right_panel.addWidget(bp_frame)

        right_panel.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content_layout.addLayout(right_panel, 1)

        # state vars
        self.reader = None
        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_plot)

        self.fs = 125
        self.display_seconds = 15
        self.buffer_size = int(self.fs * self.display_seconds)

        self.signal_buffer = []
        self.time_buffer = []
        self.start_time = None
        self.max_duration = 60

        self.ptr = 0
        self.frame_count = 0

        self.ptpp = Pan_Tompkins_Plus_Plus()
        self.latest_bpm = 0
        self.last_detect_time = time.time()
        self.last_resp_time = time.time()
        self.last_bp_time = time.time()

        # Baseline RMS dan smoothing buffers
        self.baseline_rms = None
        self.sbp_hist = deque(maxlen=3)
        self.dbp_hist = deque(maxlen=3)

        # load models (use absolute paths relative to this file)
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODELS_DIR = os.path.join(BASE_DIR, "model")
            SBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_sbp_revisi2.pkl")
            DBP_MODEL_PKL = os.path.join(MODELS_DIR, "gpr_dbp_revisi2.pkl")
            FEATURE_ORDER_JSON = os.path.join(MODELS_DIR, "features_order2.json")

            # load model objects (assume saved pipeline or model)
            self.model_sbp = joblib.load(SBP_MODEL_PKL)
            self.model_dbp = joblib.load(DBP_MODEL_PKL)

            # features order expected by model
            self.features_order = json.load(open(FEATURE_ORDER_JSON, "r"))
            print("✅ Blood Pressure models loaded successfully")
            print("[DEBUG] features_order:", self.features_order)
        except Exception as e:
            print("⚠️ Failed to load BP models:", e)
            self.model_sbp = None
            self.model_dbp = None
            self.features_order = None

    def on_connected(self):
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#1b5e20;")

    def on_disconnected(self):
        self.status_label.setText("Not Connected")
        self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#b71c1c;")

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.port_combo.addItem("Tidak ada port terdeteksi")
        else:
            for p in ports:
                self.port_combo.addItem(p.device)

    def start_stream(self):
        self.start_time = None
        # --- RESET SEMUA STATE ---
        self.last_bp_time = 0
        self.last_detect_time = 0
        self.last_resp_time = 0

        self.sbp_hist.clear()
        self.dbp_hist.clear()
        self.hr_hist.clear()
        self.hrv_hist.clear()

        # reset domain adaptation
        self.da_ready = False
        self.da_total_windows = 0
        for name in self.da_stats:
            self.da_stats[name] = {"n":0, "mean":0.0, "M2":0.0}

        # reset buffer
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.ptr = 0
        self.frame_count = 0
        
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#f9a825;")
        
        port = self.port_combo.currentText()
        mode = "Shimmer ECG" 

        if "Tidak ada" in port:
            QMessageBox.warning(self, "Error", "Tidak ada port yang bisa digunakan.")
            self.status_label.setText("Not Connected")
            self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#b71c1c;")
            return

        # Reset baseline dan history saat start
        self.baseline_rms = None
        self.sbp_hist.clear()
        self.dbp_hist.clear()

        try:
            if mode == "Shimmer ECG" and SHIMMER_AVAILABLE:
                self.reader = ShimmerReader(port)
                self.reader.error_signal.connect(self.on_shimmer_error)
            else:
                self.reader = SerialReader(port, baudrate=115200)

            self.reader.new_data.connect(self.on_new_data)
            self.reader.start()
            self.timer.start()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            self.reader.connected_signal.connect(self.on_connected)
            self.reader.disconnected_signal.connect(self.on_disconnected)
            
            self.ecg_curve.clear()
            self.hr_label.setText("--")
            self.hr_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #333;")
            self.resp_label.setText("--")
            self.resp_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #333;")
            self.bp_label.setText("--")
            self.bp_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #333;")
            self.bp_std_min_label.setText("min: --")
            self.bp_std_min_label.setStyleSheet("font-size: 10pt; color: #555;")
            self.bp_std_max_label.setText("max: --")
            self.bp_std_max_label.setStyleSheet("font-size: 10pt; color: #555;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal membuka port {port}\n\n{str(e)}")

    def on_shimmer_error(self, error_msg):
        QMessageBox.warning(self, "Shimmer Error", error_msg)
        self.stop_stream()

    def stop_stream(self):
        self.timer.stop()

        if self.reader:
            self.reader.stop()    # hanya set flag
            self.reader.wait()    # tunggu thread selesai 
            self.reader = None

        # reset GUI
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.ptr = 0
        self.frame_count = 0
        self.start_time = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Not Connected")
        self.status_label.setStyleSheet("font-size:16px; font-weight:bold; color:#b71c1c;")


    def on_new_data(self, value: float):
        now = time.time()
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time
        self.signal_buffer.append(value)
        self.time_buffer.append(t)

        max_samples = int(self.fs * self.max_duration)
        if len(self.signal_buffer) > max_samples:
            excess = len(self.signal_buffer) - max_samples
            self.signal_buffer = self.signal_buffer[excess:]
            self.time_buffer = self.time_buffer[excess:]

    @pyqtSlot()
    def update_plot(self):
        if len(self.signal_buffer) == 0:
            return

        data = np.array(self.signal_buffer)
        tdata = np.array(self.time_buffer)

        self.ecg_curve.setData(tdata, data)

        t_max = tdata[-1]
        t_min = max(0, t_max - self.display_seconds)
        self.pg_plot.setXRange(t_min, t_max)

        if self.frame_count % int(self.fs / 5) == 0:
            mask = (tdata >= t_min) & (tdata <= t_max)
            data_visible = data[mask]
            if len(data_visible) > 0:
                ymin, ymax = np.min(data_visible), np.max(data_visible)
                if ymin == ymax:
                    ymin, ymax = ymin - 0.1, ymax + 0.1
                self.pg_plot.setYRange(ymin, ymax)
        self.frame_count += 1

        now = time.time()
        if (now - self.last_detect_time) >= 0.5:
            bpm = self.compute_heart_rate(data, self.fs)
            if bpm:
                if bpm < 60:
                    color = "#FB8C00"
                elif bpm > 100:
                    color = "#FB8C00"
                else:
                    color = "#4CAF50"
                self.hr_label.setStyleSheet(f"font-size:32pt; font-weight:bold; color:{color};")
                self.latest_bpm = bpm
                self.hr_label.setText(f"{int(bpm)}")
            self.last_detect_time = now

        if (now - self.last_resp_time) >= 60:
            resp_result = self.compute_respiratory_rate_v2(data, self.fs)
            if resp_result:
                resp_rate, resp_status = resp_result
                if resp_rate < 12:
                    color = "#FB8C00"
                elif resp_rate <= 20:
                    color = "#4CAF50"
                else:
                    color = "#FB8C00"

                self.resp_label.setStyleSheet(f"font-size:32pt; font-weight:bold; color:{color};")
                self.resp_label.setText(f"{resp_rate:.1f}")
            self.last_resp_time = now

        # window 20 detik terakhir
        if len(data) >= int(FEATURE_WINDOW_SEC * self.fs):
            if time.time() - self.last_bp_time >= PREDICT_INTERVAL:
                self.last_bp_time = time.time()
                self._run_predict_from_gui(data[-int(FEATURE_WINDOW_SEC * self.fs):])

    def _run_predict_from_gui(self, sig):
        sig = np.array(sig, float)

        # sama seperti update_loop
        sig = sig - np.mean(sig)

        try:
            b, a = butter(2, [0.5/(self.fs/2), 40/(self.fs/2)], btype="band")
            sig_f = filtfilt(b, a, sig)
            bn, an = iirnotch(50/(self.fs/2), Q=30)
            sig_f = filtfilt(bn, an, sig_f)
        except:
            sig_f = sig

        sig_norm = sig_f / (np.max(np.abs(sig_f)) + 1e-9)
        r_raw = self.detector.rpeak_detection(sig_norm, self.fs)
        r_peaks = refine_rpeaks(sig_norm, r_raw, self.fs)

        self._run_predict(sig_f, r_peaks)

    def compute_heart_rate(self, signal, fs):
        try:
            r_peaks = self.ptpp.rpeak_detection(ecg=signal, fs=fs)

            corrected_peaks = []
            new_thresh = int(0.200 * fs)
            flag = 0
            for i in range(len(r_peaks)):
                if i > 0 and (r_peaks[i] - r_peaks[i-1]) < new_thresh:
                    if flag == 0:
                        flag = 1
                        continue
                corrected_peaks.append(r_peaks[i])
                flag = 0
            corrected_peaks = np.array(corrected_peaks)

            if len(corrected_peaks) > 1:
                rr_intervals = np.diff(corrected_peaks) / fs
                bpm = 60.0 / rr_intervals
                avg_bpm = np.mean(bpm)
                return avg_bpm
            else:
                return None
        except Exception as e:
            print("Pan-Tompkins++ error:", e)
            return None

    def compute_respiratory_rate_v2(self, signal, fs):
        import numpy as np
        from scipy.signal import butter, filtfilt

        try:
            # R-peak detection
            r_peaks = np.array(self.ptpp.rpeak_detection(ecg=signal, fs=fs), dtype=int)
            r_peaks = np.array(r_peaks)

            if len(r_peaks) < 5:
                return None

            # Filter 8–40 Hz 
            b, a = butter(2, [8/(fs/2), 40/(fs/2)], btype="band")
            filtered_ecg = filtfilt(b, a, signal)

            # Ekstraksi QRS window dan RMS 
            qrs_win = int(0.08 * fs)
            rms_values = []

            for r in r_peaks:
                s = max(0, r - qrs_win)
                e = min(len(filtered_ecg), r + qrs_win)

                seg = filtered_ecg[s:e]

                target_len = 2*qrs_win + 1
                if len(seg) < target_len:
                    pad = target_len - len(seg)
                    seg = np.pad(seg, (0, pad), mode='constant')

                rms_values.append(np.sqrt(np.mean(seg**2)))

            rms_values = np.array(rms_values) * 1000.0

            # RR interval
            rr_intervals = np.diff(r_peaks) / fs

            if len(rr_intervals) < 15 or len(rms_values) < 16:
                return None

            # Window mekanisme (16 & 15)
            rms_window_size = 16
            rr_window_size = 15

            num_iter = len(rms_values) - rms_window_size + 1

            resp_rates = []

            for i in range(num_iter):
                rr_w = rr_intervals[i:i+rr_window_size]
                median_rr = np.median(rr_w)

                # Frekuensi respirasi = peak FFT / median RR
                rms_w = rms_values[i:i+rms_window_size]
                n = len(rms_w)

                fft_vals = np.fft.fft(rms_w)
                freqs = np.arange(0, n//2) / n
                power = np.abs(fft_vals[:n//2])**2

                # freq range 0.083–0.667 Hz
                mask = (freqs >= 0.083) & (freqs <= 0.667)

                if not np.any(mask):
                    continue

                f_valid = freqs[mask]
                p_valid = power[mask]

                peak_idx = np.argmax(p_valid)
                f_peak = f_valid[peak_idx]

                rr_bpm = (f_peak * 60) / median_rr

                if 4 <= rr_bpm <= 40:
                    resp_rates.append(rr_bpm)

            if len(resp_rates) == 0:
                print("returned non")
                return None

            # Final median
            median_resp = float(np.median(resp_rates))

            # Klasifikasi
            if median_resp < 12:
                status = "Slow"
            elif median_resp <= 20:
                status = "Normal"
            else:
                status = "Fast"

            return round(median_resp), status

        except Exception as e:
            print("Resp rate error:", e)
            return None

    def compute_blood_pressure(self, signal, fs):
        """
        FINAL VERSION:
        - Memakai preprocessing sama seperti update_loop()
        - Refinement R-peaks sama GUI
        - FEATURE_WINDOW_SEC = 20 detik
        - Domain Adaptation + Calibration
        - SBP = mean - std, DBP = mean + std
        - Smoothing 3 window
        """
        try:
            sig = np.asarray(signal, float)
            if sig.size < int(20 * fs):
                print("[BP] ❌ Sinyal < 20 detik")
                return None, None

            # PRE-PROCESSING 
            sig = sig - np.mean(sig)

            try:
                b, a = butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
                sig_f = filtfilt(b, a, sig)

                bn, an = iirnotch(50.0 / (fs / 2), Q=30)
                sig_f = filtfilt(bn, an, sig_f)
            except Exception as e:
                print(f"[BP] Filter ERROR: {e}")
                sig_f = sig

            # R-PEAK DETECTION
            try:
                sig_norm = sig_f / (np.max(np.abs(sig_f)) + 1e-9)
                raw_peaks = self.detector.rpeak_detection(sig_norm, fs)
                r_peaks = refine_rpeaks(sig_norm, raw_peaks, fs)
            except Exception as e:
                print(f"[BP] R-peak ERROR: {e}")
                return None, None

            if len(r_peaks) < 3:
                print("[BP] ❌ R-peaks terlalu sedikit")
                return None, None

            # WINDOW 20 SECOND 
            win = int(20 * fs)
            buf = sig_f[-win:]
            r_win = r_peaks[r_peaks >= len(sig_f) - win] - (len(sig_f) - win)

            if len(r_win) < 3:
                print("[BP] ❌ R-peak di window tidak cukup")
                return None, None

            # FEATURE EXTRACTION (FeatureExtractorRevisi)
            feat = self.extractor.compute(buf, r_win)
            if feat is None:
                print("[BP] ❌ Fitur tidak lengkap")
                return None, None

            # DOMAIN ADAPTATION 
            self._update_da_stats(feat)
            feat_for_model = self._apply_domain_adapt(feat)

            # PREDIKSI GPR (mean ± std)
            sbp_mean, sbp_std, dbp_mean, dbp_std, err = self.predictor.predict(feat_for_model)
            if err or sbp_mean is None:
                print(f"[BP] ❌ Prediksi error: {err}")
                return None, None

            # fisiologi
            sbp_mean = float(np.clip(sbp_mean, 80, 200))
            dbp_mean = float(np.clip(dbp_mean, 40, 130))

            # FINAL SBP/DBP = mean - std / mean + std

            sbp = float(sbp_mean - sbp_std)
            dbp = float(dbp_mean + dbp_std)

            # SMOOTHING 3 WINDOW 
            self.sbp_hist.append(sbp)
            self.dbp_hist.append(dbp)

            sbp_final = float(np.nanmean(self.sbp_hist))
            dbp_final = float(np.nanmean(self.dbp_hist))

            return round(sbp_final, 1), round(dbp_final, 1)

        except Exception as e:
            print(f"[BP] ERROR: {e}")
            return None, None

    def _update_da_stats(self, feat):
        """Update online mean/std fitur Shimmer."""
        for name, st in self.da_stats.items():
            if name not in feat:
                continue
            val = feat[name]
            if val is None or not math.isfinite(val):
                continue
            n = st["n"] + 1
            delta = val - st["mean"]
            mean = st["mean"] + delta / n
            M2 = st["M2"] + delta * (val - mean)
            st["n"], st["mean"], st["M2"] = n, mean, M2

        self.da_total_windows += 1

        # cek apakah semua fitur punya cukup sampel dan varian
        if not self.da_ready and self.da_total_windows >= DA_MIN_SAMPLES:
            ready = True
            for name, st in self.da_stats.items():
                if st["n"] < max(3, DA_MIN_SAMPLES // 2):
                    ready = False
                    break
                if st["n"] > 1:
                    std = math.sqrt(st["M2"] / (st["n"] - 1))
                    if std < 1e-6:
                        ready = False
                        break
            self.da_ready = ready
            if self.da_ready:
                print("[INFO] Domain Adaptation siap. Fitur Shimmer akan dipetakan ke domain training.")

    def _apply_domain_adapt(self, feat):
        """Map fitur Shimmer → domain training (MITBIH_STATS)."""
        if not self.da_ready:
            return feat  # belum siap, pakai apa adanya dulu

        adapted = dict(feat)  # copy
        for name, target in MITBIH_STATS.items():
            if name not in feat:
                continue
            st = self.da_stats[name]
            if st["n"] < 2:
                continue
            mu_s = st["mean"]
            std_s = math.sqrt(st["M2"] / (st["n"] - 1))
            if std_s < 1e-6:
                continue

            mu_t = target["mean"]
            std_t = target["std"] if target["std"] > 0 else 1.0

            z = (feat[name] - mu_s) / std_s
            adapted[name] = z * std_t + mu_t

        return adapted

    # ---------- Prediction ----------
    def _run_predict(self, sig_f, r_peaks):
        win = int(FEATURE_WINDOW_SEC * FS)
        offset = len(sig_f) - win
        if offset < 0:
            return

        buf = sig_f[offset:]
        r_win = r_peaks[r_peaks >= offset] - offset
        if len(r_win) < 3:
            print(f"[WARN] R-peak terlalu sedikit dalam window (len={len(r_win)}), skip prediksi.")
            return

        feat = self.extractor.compute(buf, r_win)
        if feat is None:
            print("[WARN] Fitur tidak lengkap, skip prediksi.")
            return

        # update statistik Shimmer & lakukan domain adaptation
        self._update_da_stats(feat)
        feat_for_model = self._apply_domain_adapt(feat)

        sbp_mean, sbp_std, dbp_mean, dbp_std, err = self.predictor.predict(feat_for_model)
        if err or sbp_mean is None:
            print(f"[ERROR] Prediksi gagal: {err}")
            return

        # clipping ringan pada range fisiologis
        sbp_mean = float(np.clip(sbp_mean, 80, 200))
        dbp_mean = float(np.clip(dbp_mean, 40, 130))

        # smoothing
        self.sbp_hist.append(sbp_mean)
        self.dbp_hist.append(dbp_mean)
        self.hr_hist.append(feat.get("HR(bpm)", np.nan))
        self.hrv_hist.append(feat.get("HRV(ms)", np.nan))

        sbp_s = float(sbp_mean)
        dbp_s = float(dbp_mean)
        hr_s = float(np.nanmean(self.hr_hist))
        hrv_s = float(np.nanmean(self.hrv_hist))

        label, color = Category.label_and_color(sbp_s, dbp_s)

        self.bp_std_min_label.setText(
            f"<span style='font-size:10pt;color:#555;'>min: {sbp_mean - sbp_std:.1f} / {dbp_mean - dbp_std:.1f} </span>"
        )
        self.bp_std_max_label.setText(
            f"<span style='font-size:10pt;color:#555;'>max: {sbp_mean + sbp_std:.1f} / {dbp_mean + dbp_std:.1f} </span>"
        )
        self.bp_label.setText(
            f"<span style='font-size:32pt;font-weight:bold;color:{color};'>{int(sbp_s)}/{int(dbp_s)}</span>"
        )

        return sbp_s, dbp_s, sbp_std, dbp_std

class Category:
    @staticmethod
    def label_and_color(sbp, dbp):
        if sbp < 90 or dbp < 60:
            return "Low", "blue"
        elif 90 <= sbp <= 120 and 60 <= dbp <= 80:
            return "Normal", "green"
        elif 120 < sbp <= 139 or 80 < dbp <= 89:
            return "Elevated", "orange"
        elif sbp >= 140 or dbp >= 90:
            return "High", "red"
        else:
            return "Unknown", "gray"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECG_GUI()
    win.showMaximized()
    sys.exit(app.exec_())
