"""
=============================================================================
TUGAS BESAR KECERDASAN BUATAN (TE2514031) - SEMESTER GASAL 2026/2027
Kelompok 13: Human Activity Recognition from Smartphone Sensors
Minggu 12: EDA, Preprocessing, dan Subject-Wise Train/Val/Test Split
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Konfigurasi Path Dataset
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASET_DIR = os.path.join(
    WORKSPACE_ROOT, 
    "human+activity+recognition+using+smartphones", 
    "UCI HAR Dataset", 
    "UCI HAR Dataset"
)
OUTPUT_DIR = BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Fungsi Pembantu Load Metadata
# ---------------------------------------------------------------------------
def load_metadata():
    activity_file = os.path.join(DATASET_DIR, "activity_labels.txt")
    features_file = os.path.join(DATASET_DIR, "features.txt")
    
    activity_labels = {}
    with open(activity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                activity_labels[int(parts[0])] = parts[1]
                
    feature_names = []
    with open(features_file, "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) >= 2:
                feature_names.append(parts[1])
                
    return activity_labels, feature_names

# ---------------------------------------------------------------------------
# 3. Load 561 Feature Vectors
# ---------------------------------------------------------------------------
def load_features_dataset():
    print("[INFO] Membaca dataset 561 Fitur Rekayasa (Engineered Features)...")
    
    # Train
    X_train_path = os.path.join(DATASET_DIR, "train", "X_train.txt")
    y_train_path = os.path.join(DATASET_DIR, "train", "y_train.txt")
    sub_train_path = os.path.join(DATASET_DIR, "train", "subject_train.txt")
    
    X_train = pd.read_csv(X_train_path, sep=r"\s+", header=None).values
    y_train = pd.read_csv(y_train_path, sep=r"\s+", header=None).values.squeeze()
    sub_train = pd.read_csv(sub_train_path, sep=r"\s+", header=None).values.squeeze()
    
    # Test
    X_test_path = os.path.join(DATASET_DIR, "test", "X_test.txt")
    y_test_path = os.path.join(DATASET_DIR, "test", "y_test.txt")
    sub_test_path = os.path.join(DATASET_DIR, "test", "subject_test.txt")
    
    X_test = pd.read_csv(X_test_path, sep=r"\s+", header=None).values
    y_test = pd.read_csv(y_test_path, sep=r"\s+", header=None).values.squeeze()
    sub_test = pd.read_csv(sub_test_path, sep=r"\s+", header=None).values.squeeze()
    
    print(f"       X_train: {X_train.shape}, y_train: {y_train.shape}, sub_train: {sub_train.shape}")
    print(f"       X_test : {X_test.shape}, y_test : {y_test.shape}, sub_test : {sub_test.shape}")
    return (X_train, y_train, sub_train), (X_test, y_test, sub_test)

# ---------------------------------------------------------------------------
# 4. Load Raw Inertial Signals (9 Channels x 128 Timesteps)
# ---------------------------------------------------------------------------
def load_inertial_signals(subset="train"):
    signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    folder = os.path.join(DATASET_DIR, subset, "Inertial Signals")
    data_list = []
    
    for sig in signals:
        filename = f"{sig}_{subset}.txt"
        filepath = os.path.join(folder, filename)
        arr = pd.read_csv(filepath, sep=r"\s+", header=None).values
        data_list.append(arr)
        
    # Stack along channels: shape (N, 128, 9)
    tensor = np.stack(data_list, axis=-1)
    return tensor

# ---------------------------------------------------------------------------
# 5. Eksplorasi Data & Visualisasi (EDA)
# ---------------------------------------------------------------------------
def perform_eda(X_train, y_train, sub_train, X_test, y_test, sub_test, activity_labels):
    print("\n[INFO] Menjalankan Analisis Eksplorasi Data (EDA)...")
    
    # 5.1 Cek Missing Values
    nan_train = np.isnan(X_train).sum()
    nan_test = np.isnan(X_test).sum()
    print(f"       Missing Values di Training Set : {nan_train}")
    print(f"       Missing Values di Testing Set  : {nan_test}")
    
    # 5.2 Distribusi Partisipan (Subject)
    unique_train_subs = np.unique(sub_train)
    unique_test_subs = np.unique(sub_test)
    print(f"       Subjek di Train Set ({len(unique_train_subs)} subjek): {unique_train_subs}")
    print(f"       Subjek di Test Set  ({len(unique_test_subs)} subjek): {unique_test_subs}")
    
    # Verifikasi tidak ada overlap subjek (Pencegahan Data Leakage)
    leakage = set(unique_train_subs).intersection(set(unique_test_subs))
    if len(leakage) == 0:
        print("       [PASS] Tidak ada overlap subjek antara Train & Test (Data Leakage dicegah!).")
    else:
        print(f"       [WARNING] Ditemukan subject overlap: {leakage}")
        
    # 5.3 Distribusi Kelas Target
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    df_dist = pd.DataFrame({
        "Activity": [activity_labels[i] for i in train_counts.index],
        "Train_Count": train_counts.values,
        "Train_Pct": (train_counts.values / len(y_train) * 100).round(2),
        "Test_Count": test_counts.values,
        "Test_Pct": (test_counts.values / len(y_test) * 100).round(2)
    })
    print("\n--- Tabel Distribusi Kelas Target ---")
    print(df_dist.to_string(index=False))
    
    # Simpan plot distribusi kelas
    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(len(df_dist))
    width = 0.35
    ax.bar(x_indices - width/2, df_dist["Train_Count"], width, label="Train Set", color="#1F4E79")
    ax.bar(x_indices + width/2, df_dist["Test_Count"], width, label="Test Set", color="#E26D5C")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(df_dist["Activity"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Jumlah Sampel")
    ax.set_title("Distribusi Kelas Aktivitas Manusia (UCI HAR)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, "eda_class_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"       Plot distribusi kelas disimpan ke: {chart_path}")

# ---------------------------------------------------------------------------
# 6. Subject-Wise Train/Validation/Test Split
# ---------------------------------------------------------------------------
def split_train_val_by_subject(X_train_full, y_train_full, sub_train_full, val_subjects=None):
    """
    Membagi training set menjadi Train dan Validation secara berbasis subjek (Subject-Wise)
    agar tidak ada data windowing dari subjek yang sama bocor ke validation set.
    """
    if val_subjects is None:
        # 21 subjek di train: kita ambil 5 subjek (misal: 27, 28, 29, 30, 21) untuk validation
        val_subjects = [21, 27, 28, 29, 30]
        
    train_mask = ~np.isin(sub_train_full, val_subjects)
    val_mask = np.isin(sub_train_full, val_subjects)
    
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    sub_train = sub_train_full[train_mask]
    
    X_val = X_train_full[val_mask]
    y_val = y_train_full[val_mask]
    sub_val = sub_train_full[val_mask]
    
    print("\n[INFO] Subject-Wise Train / Validation Split:")
    print(f"       Subjek Training   : {np.unique(sub_train)} (Total: {len(X_train)} sampel)")
    print(f"       Subjek Validation : {np.unique(sub_val)} (Total: {len(X_val)} sampel)")
    
    return (X_train, y_train, sub_train), (X_val, y_val, sub_val)

# ---------------------------------------------------------------------------
# 7. Eksekusi Utama & Penyimpanan Artefak Preprocessing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print(" EKSPLORASI DATA & PREPROCESSING: UCI HAR DATASET (KELOMPOK 13)")
    print("=" * 70)
    
    activity_labels, feature_names = load_metadata()
    (X_tr_full, y_tr_full, sub_tr_full), (X_test, y_test, sub_test) = load_features_dataset()
    
    # Jalankan EDA
    perform_eda(X_tr_full, y_tr_full, sub_tr_full, X_test, y_test, sub_test, activity_labels)
    
    # Subject-wise Train/Val Split
    (X_tr, y_tr, sub_tr), (X_val, y_val, sub_val) = split_train_val_by_subject(
        X_tr_full, y_tr_full, sub_tr_full
    )
    
    # Load sampel raw inertial signals untuk demo
    print("\n[INFO] Membaca raw inertial signals tensor (train & test)...")
    raw_train = load_inertial_signals("train")
    raw_test = load_inertial_signals("test")
    print(f"       Raw Inertial Train Tensor: {raw_train.shape} (Sampel, 128 Timesteps, 9 Sinyal)")
    print(f"       Raw Inertial Test Tensor : {raw_test.shape}")
    
    # Plot sampel sinyal sensor untuk aktivitas dinamis (Walking) vs statis (Sitting)
    walking_idx = np.where(y_tr_full == 1)[0][0]
    sitting_idx = np.where(y_tr_full == 4)[0][0]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharey=True)
    axes[0].plot(raw_train[walking_idx, :, 0], label="Body Acc X", color="red")
    axes[0].plot(raw_train[walking_idx, :, 1], label="Body Acc Y", color="green")
    axes[0].plot(raw_train[walking_idx, :, 2], label="Body Acc Z", color="blue")
    axes[0].set_title("Sampel Sinyal Akselerasi Tubuh - WALKING (Dinamis)", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].plot(raw_train[sitting_idx, :, 0], label="Body Acc X", color="red")
    axes[1].plot(raw_train[sitting_idx, :, 1], label="Body Acc Y", color="green")
    axes[1].plot(raw_train[sitting_idx, :, 2], label="Body Acc Z", color="blue")
    axes[1].set_title("Sampel Sinyal Akselerasi Tubuh - SITTING (Statis)", fontweight="bold")
    axes[1].set_xlabel("Timestep (2.56 detik / 128 sample readings)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    signal_plot_path = os.path.join(PLOTS_DIR, "eda_sensor_signals.png")
    plt.savefig(signal_plot_path, dpi=300)
    plt.close()
    print(f"       Plot sinyal sensor disimpan ke: {signal_plot_path}")
    
    # Simpan partisi ter-split bebas leakage ke file biner npz agar siap dimuat cepat
    npz_path = os.path.join(OUTPUT_DIR, "preprocessed_har_data.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_tr, y_train=y_tr, sub_train=sub_tr,
        X_val=X_val, y_val=y_val, sub_val=sub_val,
        X_test=X_test, y_test=y_test, sub_test=sub_test,
        feature_names=np.array(feature_names),
        activity_ids=np.array(list(activity_labels.keys())),
        activity_names=np.array(list(activity_labels.values()))
    )
    print(f"\n[SUKSES] Seluruh data preprocessing tersimpan dalam: {npz_path}")

