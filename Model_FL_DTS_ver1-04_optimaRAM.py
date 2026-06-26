#!/usr/bin/env python3
"""
=============================================================
  DTS-FA: Dynamic Trust Score dengan Federated Aggregation
  Deteksi Serangan DDoS pada SDN Multi-Controller

  Implementasi: Disertasi Bab III — Algoritma 1 (DTS-FA)

  Dataset:
    1. CICIDS2017  → sweety18/cicids2017-full-dataset
    2. CICDDoS2019 → dhoogla/cicddos2019
    3. DDoS-SDN    → aikenkazin/ddos-sdn-dataset

  Output (file terpisah):
    loss_vs_round_<dataset>.png    — per dataset (2 panel)
    confusion_matrix_<dataset>.png — per dataset
    accuracy_vs_dataset.png
    metrics_comparison.png
    results_summary.csv

  Setup:
    pip install kaggle torch scikit-learn seaborn pandas matplotlib psutil
    kaggle.json sudah ada di /home/rikie/.kaggle/kaggle.json
=============================================================
"""

# ========================= IMPORTS =========================
import os, sys, glob, copy, gc, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ========================= KONFIGURASI DTS-FA =========================
K       = 6       # Jumlah controller (5–10)
BETA1   = 0.4     # β₁ bobot akurasi
BETA2   = 0.4     # β₂ bobot ΔL (stabilitas konvergensi)
BETA3   = 0.2     # β₃ bobot loss sesaat
DELTA   = 1e-3    # δ  ambang konvergensi
TAU     = 2       # τ  kesabaran konvergensi
T_MAX   = 200     # T_max (Algorithm 1)
EPSILON = 1e-8    # ε  stabilitas numerik

# ========================= KONFIGURASI CPU =========================
N_THREADS = os.cpu_count() or 4
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(max(1, N_THREADS // 2))
DEVICE = torch.device('cpu')   # ThinkPad T580 — no GPU

BATCH_SIZE   = 512
LOCAL_EPOCHS = 2 #Ini sama dengan TAU 
LR           = 1e-3
SAMPLE_SIZE  = 50_000

OUTPUT_DIR = 'outputs'
DATA_DIR   = 'datasets'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# Arahkan kaggle API ke lokasi kaggle.json milik user
KAGGLE_CONFIG_DIR = os.path.expanduser('/home/rikie/.kaggle')
os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_CONFIG_DIR

# ========================= KONFIGURASI DATASET =========================
# CICIDS2017: dataset ID dari sweety18 (URL /code/ adalah notebook;
#             dataset-nya ada di /datasets/sweety18/cicids2017-full-dataset)
DATASET_CONFIGS = {
    'CICIDS2017': {
        'kaggle_id'        : 'sweety18/cicids2017-full-dataset',
        'label_candidates' : [' Label', 'Label', 'label'],
        'normal_value'     : 'BENIGN',
    },
    'CICDDoS2019': {
       # 'kaggle_id'        : 'dhoogla/cicddos2019',
        'kaggle_id'        : 'kristianfrossos/cicddos2019', 
        'label_candidates' : ['Label', ' Label', 'label'],
        'normal_value'     : 'BENIGN',
    },
    'DDoS-SDN': {
        'kaggle_id'        : 'aikenkazin/ddos-sdn-dataset',
        'label_candidates' : ['label', 'Label', ' Label', 'class', 'Class'],
        'normal_value'     : None,   # auto-detect: mayoritas = normal
    },
}

# ========================= UTILITAS =========================
def mem_info() -> str:
    if not HAS_PSUTIL:
        return ""
    p     = psutil.Process(os.getpid())
    rss   = p.memory_info().rss / 1024**2
    used  = psutil.virtual_memory().used  / 1024**2
    total = psutil.virtual_memory().total / 1024**2
    return f"[RAM {rss:.0f}MB proses | {used:.0f}/{total:.0f}MB sistem]"


def download_dataset(kaggle_id: str, target_dir: str) -> str:
    """
    Download dataset dari Kaggle menggunakan kaggle API resmi.
    Membaca credentials dari KAGGLE_CONFIG_DIR yang sudah di-set.
    """
    folder_name = kaggle_id.split('/')[-1]
    dest        = os.path.join(target_dir, folder_name)

    if os.path.isdir(dest):
        csvs = glob.glob(os.path.join(dest, '**/*.csv'), recursive=True)
        if csvs:
            print(f"  [CACHE] Dataset sudah tersedia: {dest} ({len(csvs)} file CSV)")
            return dest

    print(f"  [DOWNLOAD] {kaggle_id} → {dest}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        api.dataset_download_files(kaggle_id, path=dest, unzip=True, quiet=False)
    except Exception as e:
        raise RuntimeError(
            f"Gagal download '{kaggle_id}'.\n"
            f"  Pastikan kaggle.json ada di {KAGGLE_CONFIG_DIR}\n"
            f"  Error asli: {e}"
        )
    return dest


def find_label_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'label' in c.lower() or 'class' in c.lower():
            return c
    raise ValueError(
        f"Kolom label tidak ditemukan. Kolom tersedia:\n  {df.columns.tolist()}"
    )


def load_dataset(name: str, config: dict) -> tuple:
    """
    Load dataset dari Kaggle secara chunked (hemat RAM).
    Return: (X: float32 array, y: float32 array, n_features: int)
    """
    print(f"\n{'='*55}")
    print(f"  DATASET: {name}")
    print(f"{'='*55}")

    path      = download_dataset(config['kaggle_id'], DATA_DIR)
    csv_files = sorted(glob.glob(os.path.join(path, '**/*.csv'), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
    if not csv_files:
        raise RuntimeError(f"Tidak ada file CSV ditemukan di {path}")

    print(f"  File CSV ditemukan: {len(csv_files)}")
    per_file     = max(500, SAMPLE_SIZE // max(len(csv_files), 1))
    label_col    = None
    normal_val   = config['normal_value']
    chunks_X     = []
    chunks_y     = []
    total_loaded = 0

    for fpath in csv_files:
        if total_loaded >= SAMPLE_SIZE:
            break
        try:
            tmp = pd.read_csv(fpath, nrows=per_file, low_memory=False)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fpath)}: {e}")
            continue

        # Deteksi kolom label hanya sekali di file pertama
        if label_col is None:
            label_col = find_label_col(tmp, config['label_candidates'])
            if normal_val is None:
                normal_val = str(tmp[label_col].value_counts().idxmax())
                print(f"  Auto-deteksi kelas normal: '{normal_val}'")
            print(f"  Kolom label  : '{label_col}'")
            print(f"  Kelas normal : '{normal_val}'")

        if label_col not in tmp.columns:
            continue

        yc = (tmp[label_col].astype(str).str.strip() != normal_val).astype(np.float32)
        Xc = tmp.drop(columns=[label_col]).select_dtypes(include=[np.number])
        chunks_X.append(Xc)
        chunks_y.append(yc)
        total_loaded += len(tmp)
        del tmp
        gc.collect()

    if not chunks_X:
        raise RuntimeError(f"Tidak ada data yang berhasil dimuat untuk {name}")

    df    = pd.concat(chunks_X, ignore_index=True)
    y_raw = pd.concat(chunks_y, ignore_index=True)
    del chunks_X, chunks_y
    gc.collect()

    print(f"  Total baris  : {len(df):,}")
    print(f"  Rasio attack : {y_raw.mean():.2%}")
    print(f"  {mem_info()}")

    # Bersihkan fitur
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, thresh=int(0.7 * len(df)), inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = df.loc[:, df.var() > 0]
    print(f"  Fitur bersih : {df.shape[1]}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df).astype(np.float32)
    y        = y_raw.values.astype(np.float32)
    n_feat   = X_scaled.shape[1]

    del df, y_raw
    gc.collect()
    print(f"  Load selesai. {mem_info()}")
    return X_scaled, y, n_feat


# ========================= ARSITEKTUR MODEL =========================
class DDoSDetector(nn.Module):
    """MLP kecil untuk deteksi DDoS — dioptimasi untuk CPU."""
    def __init__(self, input_dim: int,
                 hidden: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [128, 64]
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ========================= PARTISI NON-IID =========================
def partition_non_iid(X: np.ndarray, y: np.ndarray,
                       n_clients: int, alpha: float = 0.5) -> list:
    """
    Dirichlet non-IID partition. Mensimulasikan domain controller berbeda.
    Menjamin setiap client mendapat minimal MIN_SAMPLES sampel.
    """
    MIN_SAMPLES = max(BATCH_SIZE * 2, 100)
    np.random.seed(42)
    classes        = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        props = np.random.dirichlet([alpha] * n_clients)
        cuts  = (np.cumsum(props) * len(idx)).astype(int)
        for k, chunk in enumerate(np.split(idx, cuts[:-1])):
            client_indices[k].extend(chunk.tolist())

    partitions = []
    for k in range(n_clients):
        idx = np.array(client_indices[k], dtype=int)
        if len(idx) < MIN_SAMPLES:
            # Top-up dari pool global jika terlalu sedikit
            extra = np.random.choice(len(X), MIN_SAMPLES - len(idx), replace=False)
            idx   = np.concatenate([idx, extra])
        np.random.shuffle(idx)
        partitions.append((X[idx], y[idx]))

    return partitions


# ========================= PELATIHAN LOKAL =========================
def local_train(global_state: dict, data: tuple, input_dim: int) -> tuple:
    """
    Pelatihan lokal satu controller (Algorithm 1, lines 4–7).
    Return: (state_dict, val_loss, val_acc, n_train)
    """
    X_all, y_all = data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42,
        stratify=y_all if len(np.unique(y_all)) > 1 else None
    )

    model = DDoSDetector(input_dim).to(DEVICE)
    model.load_state_dict(copy.deepcopy(global_state))

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    opt       = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xv  = torch.FloatTensor(X_val)
        yv  = torch.FloatTensor(y_val)
        lg  = model(Xv)
        vloss = criterion(lg, yv).item()
        preds = (torch.sigmoid(lg) >= 0.5).numpy()
        vacc  = accuracy_score(y_val, preds)

    return model.state_dict(), vloss, vacc, len(X_tr)


# ========================= DTS-FA CORE =========================
def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Normalisasi min-max aman (Algorithm 1, lines 11–14)."""
    # Ganti inf/nan sebelum normalisasi
    arr = np.where(np.isfinite(arr), arr, 0.0)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + EPSILON)


def compute_trust_scores(accs: np.ndarray,
                          delta_losses: np.ndarray,
                          losses: np.ndarray) -> np.ndarray:
    """
    DTS — Persamaan 3.2:
    T_k = β₁·Ãcc + β₂·ΔL̃ + β₃·(1 - L̃)
    """
    return (BETA1 * minmax_norm(accs)
            + BETA2 * minmax_norm(delta_losses)
            + BETA3 * (1.0 - minmax_norm(losses)))


def federated_aggregate(state_dicts: list,
                         trust_scores: np.ndarray,
                         n_samples: np.ndarray) -> tuple:
    """
    Agregasi — Persamaan 3.1:
    α_k = (T_k·n_k) / Σ(T_i·n_i)
    w_{t+1} = Σ α_k·w_k
    """
    weights = trust_scores * n_samples
    alphas  = weights / (weights.sum() + EPSILON)

    new_state = {}
    for key in state_dicts[0]:
        new_state[key] = sum(
            alphas[k] * state_dicts[k][key].float()
            for k in range(len(state_dicts))
        )
    return new_state, alphas


def global_evaluate(model: nn.Module,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> dict:
    """Evaluasi model global pada data uji."""
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        Xt    = torch.FloatTensor(X_test)
        yt    = torch.FloatTensor(y_test)
        lg    = model(Xt)
        loss  = crit(lg, yt).item()
        preds = (torch.sigmoid(lg) >= 0.5).numpy()
    return {
        'accuracy' : accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall'   : recall_score(y_test, preds, zero_division=0),
        'f1'       : f1_score(y_test, preds, zero_division=0),
        'loss'     : loss,
        'cm'       : confusion_matrix(y_test, preds),
    }


# ========================= MAIN LOOP DTS-FA =========================
def run_dtsfa(X: np.ndarray, y: np.ndarray,
               input_dim: int, ds_name: str) -> dict:
    """
    Algoritma DTS-FA lengkap (Algorithm 1).
    Dua kondisi stop:
      Kondisi 1: |L_global^t - L_global^{t-1}| < δ selama τ round → konvergen
      Kondisi 2: t ≥ T_max → batas maksimum
    """
    print(f"\n{'#'*58}")
    print(f"#  DTS-FA — {ds_name}")
    print(f"#  K={K}  β=({BETA1},{BETA2},{BETA3})")
    print(f"#  δ={DELTA}  τ={TAU}  T_max={T_MAX}")
    print(f"{'#'*58}")

    # Split test set global
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Partisi non-IID ke K controller
    parts = partition_non_iid(X_tr, y_tr, K, alpha=0.5)
    print("\n  Distribusi data per controller:")
    for k, (Xk, yk) in enumerate(parts):
        print(f"    C{k+1:02d}: {len(Xk):6,} sampel | "
              f"attack={yk.mean():.1%}")

    # Inisialisasi model global (Algorithm 1, line 1)
    gmodel = DDoSDetector(input_dim).to(DEVICE)
    gstate = copy.deepcopy(gmodel.state_dict())

    # ── FIX BUG #3: prev_losses diinisialisasi None, bukan np.inf ──
    # np.inf menyebabkan delta_losses = inf - loss = inf,
    # lalu minmax_norm([inf,inf,...]) = nan/nan = NaN → crash
    prev_losses   = {k: None for k in range(K)}
    L_global_prev = None   # FIX BUG #4: sama, hindari inf di delta_L_global

    cnt_conv      = 0
    T_star        = T_MAX
    converge_cond = 2   # default = Kondisi 2 (T_max)

    history = {
        'round'         : [],
        'global_loss'   : [],
        'delta_L_global': [],
        'cnt_conv'      : [],
        'accuracy'      : [],
        'precision'     : [],
        'recall'        : [],
        'f1'            : [],
    }

    # ══════════════════════════════════════════════════════
    #  LOOP UTAMA (Algorithm 1, lines 2–31)
    # ══════════════════════════════════════════════════════
    for t in range(1, T_MAX + 1):

        # ── Pelatihan lokal (lines 4–7) ──
        local_states, local_losses, local_accs, local_n = [], [], [], []
        for k in range(K):
            sd, lk, ak, nk = local_train(gstate, parts[k], input_dim)
            local_states.append(sd)
            local_losses.append(lk)
            local_accs.append(ak)
            local_n.append(nk)

        losses = np.array(local_losses)
        accs   = np.array(local_accs)
        n_arr  = np.array(local_n, dtype=float)

        # ── Hitung ΔL per controller (lines 8–9) ──
        # Round pertama: prev_losses[k] = None → ΔL = 0 (tidak ada referensi)
        delta_losses = np.array([
            0.0 if prev_losses[k] is None else prev_losses[k] - local_losses[k]
            for k in range(K)
        ])

        # ── Hitung DTS (lines 15–18) ──
        T_scores = compute_trust_scores(accs, delta_losses, losses)

        # ── Agregasi federasi (lines 19–21) ──
        gstate, alphas = federated_aggregate(local_states, T_scores, n_arr)
        gmodel.load_state_dict(gstate)

        # ── Global loss (line 22) ──
        L_global_t = float(np.mean(losses))

        # Update prev_losses
        for k in range(K):
            prev_losses[k] = local_losses[k]

        # ── Evaluasi global ──
        ev = global_evaluate(gmodel, X_test, y_test)

        # ── Kondisi 1: hitung |L_global^t - L_global^{t-1}| (lines 22–25) ──
        # Round pertama: L_global_prev = None → delta = 0.0 (skip cek)
        if L_global_prev is None:
            delta_L_global = 0.0
            cnt_conv       = 0
        else:
            delta_L_global = abs(L_global_t - L_global_prev)
            if delta_L_global < DELTA:
                cnt_conv += 1
            else:
                cnt_conv = 0
        L_global_prev = L_global_t

        # Simpan history
        history['round'].append(t)
        history['global_loss'].append(L_global_t)
        history['delta_L_global'].append(delta_L_global)
        history['cnt_conv'].append(cnt_conv)
        history['accuracy'].append(ev['accuracy'])
        history['precision'].append(ev['precision'])
        history['recall'].append(ev['recall'])
        history['f1'].append(ev['f1'])

        # Log per round
        print(f"  Round {t:3d} | "
              f"loss={L_global_t:.4f} | "
              f"acc={ev['accuracy']:.4f} | "
              f"f1={ev['f1']:.4f} | "
              f"cnt_conv={cnt_conv}/{TAU} "
              f"|ΔL|={delta_L_global:.2e} "
              f"(δ={DELTA:.0e}) "
              f"{mem_info()}")

        # Bebaskan state lokal
        del local_states
        gc.collect()

        # ── Kondisi 1: berhenti jika konvergen (lines 26–28) ──
        if cnt_conv >= TAU:
            print(f"\n  [STOP — Kondisi 1] "
                  f"|ΔL_global| < δ selama {TAU} round berturut-turut")
            print(f"  T* = {t}  ← Kondisi 1: KONVERGEN")
            T_star        = t
            converge_cond = 1
            break

        # ── Kondisi 2: berhenti jika t = T_max (lines 29–31) ──
        if t >= T_MAX:
            print(f"\n  [STOP — Kondisi 2] T_max={T_MAX} tercapai")
            print(f"  T* = {t}  ← Kondisi 2: T_MAX")
            T_star        = t
            converge_cond = 2
            break

    # Evaluasi final
    final_ev = global_evaluate(gmodel, X_test, y_test)
    print(f"\n  {'─'*45}")
    print(f"  Hasil Final: {ds_name}  (T*={T_star}, "
          f"stop=Kondisi {'1:konvergen' if converge_cond==1 else '2:T_max'})")
    print(f"  Accuracy : {final_ev['accuracy']:.4f}")
    print(f"  Precision: {final_ev['precision']:.4f}")
    print(f"  Recall   : {final_ev['recall']:.4f}")
    print(f"  F1-Score : {final_ev['f1']:.4f}")
    print(f"  {'─'*45}")

    return {
        'dataset'      : ds_name,
        'history'      : history,
        'final'        : final_ev,
        'T_star'       : T_star,
        'converge_cond': converge_cond,
        'y_test'       : y_test,
    }


# ========================= VISUALISASI =========================
def plot_loss_vs_round(result: dict) -> str:
    """
    2-panel: (atas) Loss + F1 vs round
             (bawah) |ΔL_global| vs δ — visualisasi Kondisi 1
    """
    name = result['dataset']
    h    = result['history']

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={'hspace': 0.06, 'height_ratios': [3, 2]}
    )

    rounds = h['round']

    # Panel atas: Loss & F1
    ax1.plot(rounds, h['global_loss'], color='#1565C0', lw=2,
             marker='o', ms=3, label='Global Loss (L_global)')
    ax1.plot(rounds, h['f1'],          color='#2E7D32', lw=1.5,
             marker='s', ms=2, ls='--', label='F1-Score')

    cond_label = ('Kondisi 1: Konvergen'
                  if result['converge_cond'] == 1 else 'Kondisi 2: T_max')
    ax1.axvline(result['T_star'], color='red', ls=':', lw=2,
                label=f"T*={result['T_star']} ({cond_label})")

    ax1.set_ylabel('Nilai', fontsize=11)
    ax1.set_title(
        f'Loss vs Communication Round — {name}\n'
        f'DTS-FA  |  K={K} Controller  |  β=({BETA1},{BETA2},{BETA3})',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel bawah: |ΔL_global| vs δ
    # Clip ke [1e-10, inf) agar semilogy aman
    dl_safe = [max(v, 1e-10) for v in h['delta_L_global']]
    ax2.semilogy(rounds, dl_safe, color='#7B1FA2', lw=1.8,
                 marker='^', ms=3,
                 label='|L_global^t − L_global^{t−1}|')
    ax2.axhline(DELTA, color='orange', ls='--', lw=1.5,
                label=f'δ = {DELTA:.0e}  (ambang Kondisi 1)')
    ax2.fill_between(rounds, dl_safe, DELTA,
                     where=[v < DELTA for v in dl_safe],
                     alpha=0.25, color='orange',
                     label='|ΔL| < δ  (cnt_conv naik)')
    ax2.axvline(result['T_star'], color='red', ls=':', lw=2)
    ax2.set_xlabel('Communication Round', fontsize=11)
    ax2.set_ylabel('|ΔL_global|  (log scale)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'loss_vs_round_{name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_confusion_matrix(result: dict) -> str:
    """Confusion Matrix dengan jumlah dan persentase per baris."""
    name = result['dataset']
    cm   = result['final']['cm']
    pct  = cm / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    ann  = np.array([[f"{cm[i,j]}\n({pct[i,j]:.1f}%)"
                      for j in range(2)] for i in range(2)])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'DDoS'],
                yticklabels=['Normal', 'DDoS'],
                linewidths=0.5, annot_kws={'size': 13})
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Label Sebenarnya', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}\n'
                 f'DTS-FA  |  K={K} Controller',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_accuracy_vs_dataset(results: list) -> str:
    names  = [r['dataset'] for r in results]
    vals   = [r['final']['accuracy'] for r in results]
    colors = ['#1565C0', '#2E7D32', '#E65100'][:len(names)]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(names, vals, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Perbandingan Accuracy Antar Dataset\n'
                 f'DTS-FA  |  K={K} Controller',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'accuracy_vs_dataset.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_metrics_comparison(results: list) -> str:
    names   = [r['dataset'] for r in results]
    metrics = {
        'Accuracy' : [r['final']['accuracy']  for r in results],
        'Precision': [r['final']['precision'] for r in results],
        'Recall'   : [r['final']['recall']    for r in results],
        'F1-Score' : [r['final']['f1']        for r in results],
    }
    colors = ['#1565C0', '#2E7D32', '#F57C00', '#6A1B9A']
    x, w   = np.arange(len(names)), 0.18

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (label, vals) in enumerate(metrics.items()):
        offset = (i - 1.5) * w
        bars   = ax.bar(x + offset, vals, w, label=label,
                        color=colors[i], edgecolor='white', lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Perbandingan Metrik Evaluasi Antar Dataset\n'
                 f'DTS-FA  |  K={K}  |  β=({BETA1},{BETA2},{BETA3})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def save_summary(results: list) -> str:
    rows = [{
        'Dataset'        : r['dataset'],
        'Accuracy'       : round(r['final']['accuracy'],  4),
        'Precision'      : round(r['final']['precision'], 4),
        'Recall'         : round(r['final']['recall'],    4),
        'F1-Score'       : round(r['final']['f1'],        4),
        'T* (round)'     : r['T_star'],
        'Stop Condition' : ('Konvergen' if r['converge_cond'] == 1 else 'T_max'),
        'K'              : K,
    } for r in results]

    df  = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    df.to_csv(out, index=False)
    print(f"\n{'='*55}")
    print("  RINGKASAN HASIL EKSPERIMEN")
    print('='*55)
    print(df.to_string(index=False))
    print('='*55)
    return out


# ========================= ENTRY POINT =========================
def main():
    print('='*58)
    print('  DTS-FA: Dynamic Trust Score + Federated Aggregation')
    print('  DDoS Detection — SDN Multi-Controller')
    print('='*58)
    print(f"  Device   : {DEVICE}  ({N_THREADS} CPU threads)")
    print(f"  K        : {K} controllers")
    print(f"  β        : ({BETA1}, {BETA2}, {BETA3})")
    print(f"  T_max    : {T_MAX}  |  δ={DELTA}  |  τ={TAU}")
    print(f"  {mem_info()}\n")

    # Cek dependensi
    missing = []
    for pkg in ['kaggle', 'torch', 'sklearn', 'pandas',
                'numpy', 'matplotlib', 'seaborn']:
        try:
            __import__(pkg if pkg != 'sklearn' else 'sklearn')
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[INSTALL] {' '.join(missing)}")
        os.system(f"{sys.executable} -m pip install {' '.join(missing)} -q")

    # Verifikasi kaggle.json
    kj = os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json')
    if not os.path.exists(kj):
        print(f"\n[ERROR] kaggle.json tidak ditemukan di {kj}")
        print("  Unduh dari: https://www.kaggle.com/settings → API → Create Token")
        sys.exit(1)
    print(f"  ✓ kaggle.json ditemukan di {kj}")

    all_results = []

    for ds_name, config in DATASET_CONFIGS.items():
        try:
            X, y, n_feat = load_dataset(ds_name, config)
            result       = run_dtsfa(X, y, n_feat, ds_name)
            all_results.append(result)

            p1 = plot_loss_vs_round(result)
            p2 = plot_confusion_matrix(result)
            print(f"  Plot: {p1}")
            print(f"  Plot: {p2}")

            del X, y         # Bebaskan RAM sebelum dataset berikutnya
            gc.collect()

        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {ds_name}: {exc}")
            traceback.print_exc()
            print(f"  → Melanjutkan ke dataset berikutnya...\n")

    if not all_results:
        print("\n[FATAL] Tidak ada eksperimen yang berhasil.")
        return

    p3 = plot_accuracy_vs_dataset(all_results)
    p4 = plot_metrics_comparison(all_results)
    p5 = save_summary(all_results)
    print(f"\n  {p3}\n  {p4}\n  {p5}")
    print(f"\n{'='*58}")
    print("  ✓ Semua eksperimen selesai.")
    print(f"{'='*58}")


if __name__ == '__main__':
    main()