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
from datetime import datetime
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


# jumlah minimal window prediksi untuk kalibrasi domain Shimmer
DA_MIN_SAMPLES = 15 


# ============================================================
# KELAS EKSTRAKTOR FITUR RANDOM FOREST (10 Fitur Skripsi)
# ============================================================
class FeatureExtractorRF:
    def __init__(self, fs):
        self.fs = fs

    def compute(self, ecg, r_peaks):
        fs = self.fs
        N = len(ecg)
        r_peaks = np.asarray([p for p in r_peaks if 0 < p < N - 1], int)
        if N < fs * 3 or len(r_peaks) < 3:
            return None

        # 1. RMS (Root Mean Square)
        rms = float(np.sqrt(np.mean(ecg ** 2)))
        
        # 2. HR & HRV
        rr = np.diff(r_peaks) / fs
        if rr.size == 0: return None
        mean_rr = float(np.mean(rr))
        hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
        hrv = float(np.std(rr) * 1000.0)

        # 3. QT & TQ Estimation
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
        mean_qt = float(np.mean(qt_arr)) if qt_arr.size else 0.0
        mean_tq = float(np.mean(tq_arr)) if tq_arr.size else 0.0
        
        # 4. QTc Bazett
        qtc_bazett = float(mean_qt / np.sqrt(mean_rr)) if mean_qt > 0 and mean_rr > 0 else 0.0

        # 5. Hjorth Parameters
        diff1, diff2 = np.diff(ecg), np.diff(np.diff(ecg))
        var0, var1, var2 = np.var(ecg), np.var(diff1), np.var(diff2)
        hj_mob = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
        hj_comp = float(np.sqrt(var2 / var1)) if var1 > 0 else 0.0

        # 6. SDI & SDIn
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_qt_tq = qt_arr / tq_arr if qt_arr.size and tq_arr.size else []
            ratio_qt_rr = qt_arr / rr[:len(qt_arr)] if rr.size else []
        sdi_qt_tq = float(np.nanstd(ratio_qt_tq)) if len(ratio_qt_tq) else 0.0
        sdin_qt_rr = float(np.nanstd(ratio_qt_rr)) if len(ratio_qt_rr) else 0.0

        # Output 10 Fitur (Sesuai urutan training: Mean_QT, TQ, QTc_Bazett, Hjorth_Mobility, Hjorth_Complexity, RMS, SDI, SDIn, HR, HRV)
        # HR ada di indeks 8, HRV di indeks 9
        return np.array([[
            mean_qt, mean_tq, qtc_bazett, hj_mob, hj_comp, 
            rms, sdi_qt_tq, sdin_qt_rr, hr, hrv
        ]])


# ============================================================
# KELAS PREDIKTOR RANDOM FOREST (Muat model 10 fitur)
# ============================================================
class PredictorRF:
    def __init__(self):
        try:
            # Ganti dengan nama model yang dihasilkan oleh trainmodel.py yang baru
            self.model_sbp = joblib.load(os.path.join("model", "rf_sbp_N.pkl"))
            self.model_dbp = joblib.load(os.path.join("model", "rf_dbp_N.pkl"))
            self.feat_scaler = joblib.load(os.path.join("model", "feat_scaler.pkl"))
            print("✅ Model Random Forest dan Scaler berhasil dimuat!")
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            self.model_sbp, self.model_dbp, self.feat_scaler = None, None, None

    def predict(self, features):
        if self.model_sbp is None:
            return None, None, None, None, "Model belum dimuat"
        try:
            feats_scaled = self.feat_scaler.transform(features)
            sbp = self.model_sbp.predict(feats_scaled)[0]
            dbp = self.model_dbp.predict(feats_scaled)[0]
            
            # Nilai standar deviasi statis
            sbp_std, dbp_std = 2.7, 2.7 
            
            return sbp, sbp_std, dbp, dbp_std, None
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
        self.extractor = FeatureExtractorRF(FS)
        self.predictor = PredictorRF()

        self.sbp_hist = deque(maxlen=3)
        self.dbp_hist = deque(maxlen=3)
        self.hr_hist = deque(maxlen=3)
        self.hrv_hist = deque(maxlen=3)

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

    from datetime import datetime # (Pastikan datetime sudah di-import di atas file)

    def _run_predict_from_gui(self, sig):
        ecg_raw = np.array(sig, dtype=float)
        
        # 1. Preprocessing Sesuai Skripsi (Bandpass + Notch)
        ecg_clean = ecg_raw - np.mean(ecg_raw)
        b, a = butter(2, [0.5 / (self.fs / 2), 40.0 / (self.fs / 2)], btype="band")
        ecg_clean = filtfilt(b, a, ecg_clean)
        bn, an = iirnotch(50.0 / (self.fs / 2), Q=30)
        ecg_clean = filtfilt(bn, an, ecg_clean)
        
        # 2. Deteksi R-Peaks (Menggunakan Pan-Tompkins++)
        try:
            r_peaks = self.ptpp.rpeak_detection(ecg_clean, self.fs) # Gunakan self.ptpp, bukan self.detector
            r_peaks = np.array(r_peaks, int)
        except Exception as e:
            print(f"[ERROR] Deteksi R-Peak gagal: {e}")
            return
            
        if len(r_peaks) < 3:
            return
        
        # 3. Ekstraksi Fitur (10 Fitur Skripsi)
        feats = self.extractor.compute(ecg_clean, r_peaks)
        if feats is None:
            return
            
        # Ambil HR dan HRV dari feats array untuk tabel log (jika diperlukan nanti)
        #hr_val = feats[0][8]
        #hrv_val = feats[0][9]
        
        # 4. Prediksi Tekanan Darah (Random Forest)
        sbp_mean, sbp_std, dbp_mean, dbp_std, err = self.predictor.predict(feats)
        if err or sbp_mean is None:
            print(f"[ERROR] Prediksi BP gagal: {err}")
            return
            
        # 5. Batasi nilai (Clipping) & Smoothing
        sbp_mean = float(np.clip(sbp_mean, 80, 200))
        dbp_mean = float(np.clip(dbp_mean, 40, 130))

        self.sbp_hist.append(sbp_mean)
        self.dbp_hist.append(dbp_mean)

        sbp_s = float(np.nanmean(self.sbp_hist))
        dbp_s = float(np.nanmean(self.dbp_hist))

        # 6. Update Tampilan Label di GUI
        label, color = Category.label_and_color(sbp_s, dbp_s)

        # Update teks tanpa HTML span agar bounding box dihitung dengan benar oleh PyQt
        self.bp_std_min_label.setText(f"min: {sbp_mean - sbp_std:.1f} / {dbp_mean - dbp_std:.1f}")
        self.bp_std_max_label.setText(f"max: {sbp_mean + sbp_std:.1f} / {dbp_mean + dbp_std:.1f}")
        
        # Ubah warna label secara dinamis melalui styleSheet, bukan melalui HTML span
        self.bp_label.setStyleSheet(f"font-size: 36pt; font-weight: bold; color: {color};")
        self.bp_label.setText(f"{int(sbp_s)}/{int(dbp_s)}")
        
  
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
