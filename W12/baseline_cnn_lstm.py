"""
=============================================================================
TUGAS BESAR KECERDASAN BUATAN (TE2514031) - SEMESTER GASAL 2026/2027
Kelompok 13: Human Activity Recognition from Smartphone Sensors
Minggu 12: Baseline Neural Network (Model A: CNN-LSTM)
=============================================================================
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Konfigurasi Device & Path
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "preprocessed_har_data.npz")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "baseline_cnn_lstm.pt")
METRICS_SAVE_PATH = os.path.join(BASE_DIR, "baseline_metrics.json")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Definisi Arsitektur Model A: CNN-LSTM
# ---------------------------------------------------------------------------
class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_features=561, num_classes=6, cnn_filters=64, lstm_units=64, dropout_rate=0.3):
        super(BaselineCNNLSTM, self).__init__()
        
        # 1D Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM Block
        # Panjang sequence setelah pooling = num_features // 2
        self.lstm = nn.LSTM(
            input_size=cnn_filters, 
            hidden_size=lstm_units, 
            num_layers=1, 
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        # Input x: (batch_size, num_features) -> Reshape ke (batch_size, 1, num_features)
        x = x.unsqueeze(1)
        
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x) # Shape: (batch_size, cnn_filters, seq_len)
        
        # Permute untuk LSTM: (batch_size, seq_len, cnn_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM Temporal Processing
        lstm_out, (hn, _) = self.lstm(x)
        
        # Ambil representasi waktu terakhir (hn: 1, batch, lstm_units)
        last_hidden = hn[-1]
        
        # Classification
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------------------------
# 3. Fungsi Training & Evaluasi
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    total = len(all_labels)
    eval_loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return eval_loss, accuracy, precision, recall, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ---------------------------------------------------------------------------
# 4. Eksekusi Utama Model A
# ---------------------------------------------------------------------------
def run_baseline_experiment(epochs=15, batch_size=64, learning_rate=0.001):
    print("=" * 70)
    print(" TRAINING MODEL A: BASELINE CNN-LSTM (TANPA GA & TANPA PSO)")
    print(f" Device yang digunakan: {DEVICE}")
    print("=" * 70)
    
    # 4.1 Load Data
    data = np.load(DATA_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"] - 1 # Shift index label 1..6 menjadi 0..5
    X_val = data["X_val"]
    y_val = data["y_val"] - 1
    X_test = data["X_test"]
    y_test = data["y_test"] - 1
    activity_names = list(data["activity_names"])
    
    num_features = X_train.shape[1]
    num_classes = len(activity_names)
    print(f"[INFO] Jumlah Fitur Input : {num_features}")
    print(f"[INFO] Jumlah Kelas Target: {num_classes} ({activity_names})")
    print(f"[INFO] Jumlah Data Train  : {X_train.shape[0]} sampel")
    print(f"[INFO] Jumlah Data Val    : {X_val.shape[0]} sampel")
    print(f"[INFO] Jumlah Data Test   : {X_test.shape[0]} sampel")
    
    # Konversi ke PyTorch Dataset & DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4.2 Inisialisasi Model, Loss, Optimizer
    model = BaselineCNNLSTM(num_features=num_features, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    best_val_f1 = 0.0
    
    # 4.3 Training Loop
    print("\n[INFO] Memulai Proses Pelatihan Model A...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, _, _, _ = evaluate(model, val_loader, criterion)
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% || "
              f"Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc*100:.2f}% | Val F1: {vl_f1*100:.2f}%")
        
        # Simpan checkpoint model terbaik berdasarkan val_f1
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_features": int(num_features),
                "num_classes": int(num_classes),
                "best_val_f1": float(best_val_f1),
                "activity_names": [str(x) for x in activity_names]
            }, MODEL_SAVE_PATH)
            
    print(f"\n[INFO] Checkpoint model terbaik disimpan di: {MODEL_SAVE_PATH}")
    
    # 4.4 Evaluasi Akhir pada Isolated Test Set (Bebas Leakage)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels, probs = evaluate(
        model, test_loader, criterion
    )
    
    print("\n" + "=" * 50)
    print(" HASIL EVALUASI MODEL A (TEST SET BEBAS LEAKAGE)")
    print("=" * 50)
    print(f" Test Accuracy  : {test_acc * 100:.2f}%")
    print(f" Test Precision : {test_prec * 100:.2f}%")
    print(f" Test Recall    : {test_rec * 100:.2f}%")
    print(f" Test Macro F1  : {test_f1 * 100:.2f}%")
    print(f" Test Loss      : {test_loss:.4f}")
    
    # 4.5 Plot Confusion Matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=activity_names,
        yticklabels=activity_names,
        title="Confusion Matrix - Model A (Baseline CNN-LSTM)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot Confusion Matrix disimpan ke: {cm_path}")
    
    # Plot Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history["train_loss"], label="Train Loss", color="#1F4E79")
    ax1.plot(history["val_loss"], label="Val Loss", color="#E26D5C")
    ax1.set_title("Loss Curve (Model A)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.plot([acc*100 for acc in history["train_acc"]], label="Train Acc", color="#1F4E79")
    ax2.plot([acc*100 for acc in history["val_acc"]], label="Val Acc", color="#E26D5C")
    ax2.set_title("Accuracy Curve (Model A)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    curve_path = os.path.join(PLOTS_DIR, "baseline_training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot Training Curve disimpan ke: {curve_path}")
    
    # Simpan metrik ke JSON untuk kebutuhan dashboard Streamlit
    metrics_summary = {
        "model_name": "Model A (Baseline CNN-LSTM)",
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1),
        "loss": float(test_loss),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_features": num_features,
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[INFO] Ringkasan metrik disimpan ke: {METRICS_SAVE_PATH}")
    print("[SUKSES] Seluruh tahapan Model A Baseline selesai!")

if __name__ == "__main__":
    run_baseline_experiment(epochs=12, batch_size=64, learning_rate=0.001)
