import scipy.io
import pandas as pd
import numpy as np
import gc # Garbage collector untuk menghemat RAM

print("⏳ Memuat file part_1.mat...")
mat_data = scipy.io.loadmat('part_1.mat')
records = mat_data['p'][0]

all_data = []
total_patients = len(records)
print(f"📊 Ditemukan {total_patients} rekaman pasien. Memulai ekstraksi...")

for i in range(total_patients): 
    data_pasien = records[i]
    # Baris 1: ABP, Baris 2: ECG (Sesuai deskripsi UCI)
    temp_df = pd.DataFrame({
        'ABP': data_pasien[1, :],
        'ECG': data_pasien[2, :]
    })
    all_data.append(temp_df)
    
    if (i+1) % 10 == 0:
        print(f"   -> {i+1}/{total_patients} pasien diproses...")

# Kosongkan memory dari mat_data
del mat_data
del records
gc.collect()

print("⏳ Menggabungkan data (bisa memakan waktu)...")
df_final = pd.concat(all_data, ignore_index=True)
df_final.to_csv('df.csv', index=False)
print(f"✅ df.csv berhasil dibuat dengan total {len(df_final)} baris data.")